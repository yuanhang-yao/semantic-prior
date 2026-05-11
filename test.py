import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.dataset import *
from utils.metrics import PD_FA_2, SamplewiseSigmoidMetric, SigmoidMetric
from models.alclnet import ALCLNet
from models.mlclnet import MLCLNet
from models.teacher import DINOCNNTeacher, FrozenDINOBackbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--model", type=str, default="MLCL", choices=("MLCL", "ALCL"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    return parser.parse_args()


class PredictionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, names):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.names = list(names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, name)).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(float)
        image, mask, h, w = pad_to_multiple(image, mask, multiple=32)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"], name, h, w, DINO_TRANSFORM(image=image)["image"]


def predict_logits(img, net, img_dino=None, batch_size=16):
    b, c, h, w = img.shape
    patch_size = 1024
    if h > patch_size and w > patch_size:
        img_unfold = F.unfold(img, kernel_size=patch_size, stride=patch_size)
        img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        dino_unfold = None
        if img_dino is not None:
            dino_unfold = F.unfold(img_dino, kernel_size=patch_size, stride=patch_size)
            dino_unfold = dino_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        patch_num = img_unfold.size(1)
        preds = []
        for i in range(0, patch_num, batch_size):
            img_batch = img_unfold[:, i : min(i + batch_size, patch_num)].reshape(-1, c, patch_size, patch_size)
            if dino_unfold is None:
                pred_batch = net(img_batch.float())
            else:
                dino_batch = dino_unfold[:, i : min(i + batch_size, patch_num)].reshape(-1, c, patch_size, patch_size)
                pred_batch = net(img_batch.float(), dino_batch.float())
            preds.append(pred_batch)
        preds_unfold = torch.cat(preds, dim=0).permute(1, 2, 3, 0)
        preds_unfold = preds_unfold.reshape(b, -1, patch_num)
        return F.fold(preds_unfold, kernel_size=patch_size, stride=patch_size, output_size=(h, w))
    return net(img, img_dino) if img_dino is not None else net(img)


def build_student(model_name):
    if model_name == "MLCL":
        return MLCLNet(in_channels=3)
    elif model_name == "ALCL":
        return ALCLNet(in_channels=3)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device(args.device)

    paths = build_paths(args)
    train_names = load_sample_names(paths["train_split"])
    val_names = load_sample_names(paths["val_split"])
    test_names = load_sample_names(paths["test_split"])
    cal_mean, cal_std = calculate_mean_std(
        [
            (paths["train_source_img"], train_names),
            (paths["val_source_img"], val_names),
        ]
    )
    _, eval_transform = build_transforms(mean=cal_mean, std=cal_std)

    loader = DataLoader(
        PredictionDataset(paths["test_source_img"], paths["test_source_label"], eval_transform, test_names),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    is_teacher = any(key.startswith("base.") or key.startswith("modulator.") for key in state_dict)

    student = build_student(args.model)
    if is_teacher:
        model = DINOCNNTeacher(
            dino_backbone=FrozenDINOBackbone("facebook/dinov3-vits16plus-pretrain-lvd1689m", device=device),
            base_model=student,
            stage_channels=list(student.stage_channels),
            hidden_dim=128,
            residual_gamma=True,
        )
    else:
        model = student

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch. missing={missing}, unexpected={unexpected}")

    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    saved = 0
    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    fa_pd_metric = PD_FA_2(1)
    with torch.no_grad():
        for image, mask, name, h, w, image_dino in loader:
            image = image.to(device)
            mask = mask.unsqueeze(1).to(device)
            image_dino = image_dino.to(device)
            logits = predict_logits(image, model, image_dino if is_teacher else None)
            logits = logits[:, :, : int(h[0]), : int(w[0])]
            mask = mask[:, :, : int(h[0]), : int(w[0])]
            probs = torch.sigmoid(logits)
            iou_metric.update(probs, mask)
            niou_metric.update(probs, mask)
            fa_pd_metric.update(probs, mask)
            pred = probs[0, 0].cpu().numpy()
            pred = np.where(pred > 0.5, 255, 0).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output_dir, name[0]), pred)
            saved += 1

    print(f"Saved {saved} predictions to: {os.path.abspath(args.output_dir)}")
    _, iou = iou_metric.get()
    _, niou = niou_metric.get()
    fa, pd = fa_pd_metric.get()
    print(f"IoU: {iou:.4f}, nIoU: {niou:.4f}, PD: {pd:.4f}, FA: {fa:.8f}")


if __name__ == "__main__":
    main()
