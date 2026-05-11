import os
import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from args import parse_args
from utils.dataset import *
from models.alclnet import ALCLNet
from models.mlclnet import MLCLNet
from models.teacher import DINOCNNTeacher, FrozenDINOBackbone
from utils.gn_bilevel import Architect
from utils.losses import inner_loss, outer_loss
from utils.mask_update import *
from utils.metrics import PD_FA_2, SamplewiseSigmoidMetric, SigmoidMetric


def build_cluster_map_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cluster metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "cluster_id" not in df.columns:
        raise ValueError("[Cluster] 'cluster_id' column is missing.")
    image_names = df["image_name"].astype(str).tolist()
    cluster_ids = df["cluster_id"].astype(int).tolist()
    cluster_map = {os.path.basename(name): int(cid) for name, cid in zip(image_names, cluster_ids)}
    unique_ids = sorted(set(cluster_ids))
    label2idx = {label: idx for idx, label in enumerate(unique_ids)}
    return {name: label2idx[cid] for name, cid in cluster_map.items()}, len(label2idx)

class ArchModelGN(torch.nn.Module):
    def __init__(self, student, teacher, alpha, cluster_map, device):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.cluster_map = cluster_map
        self.device = device

    def parameters(self):
        return self.student.parameters()

    def arch_parameters(self):
        return list(self.student.parameters()) + [self.alpha]

    def _loss(self, batch, _unused, lamda, latency):
        return outer_loss(self.student, self.teacher, self.alpha, batch, self.cluster_map, self.device)


def test_pred(img, net, img_dino=None, batch_size=16):
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
        preds_list = []
        for i in range(0, patch_num, batch_size):
            batch_patches = img_unfold[:, i:min(i + batch_size, patch_num)].reshape(-1, c, patch_size, patch_size)
            if dino_unfold is None:
                batch_preds = net.forward(batch_patches.float())
            else:
                batch_dino = dino_unfold[:, i:min(i + batch_size, patch_num)].reshape(-1, c, patch_size, patch_size)
                batch_preds = net.forward(batch_patches.float(), batch_dino.float())
            preds_list.append(batch_preds)
        preds_unfold = torch.cat(preds_list, dim=0).permute(1, 2, 3, 0)
        preds_unfold = preds_unfold.reshape(b, -1, patch_num)
        return F.fold(preds_unfold, kernel_size=patch_size, stride=patch_size, output_size=(h, w))
    return net.forward(img, img_dino) if img_dino is not None else net.forward(img)


def evaluate(loader, model, device, teacher=False):
    model.eval()
    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    fa_pd_metric = PD_FA_2(1)
    with torch.no_grad():
        for x, y, h, w, x_dino in loader:
            x = x.to(device).detach()
            y = y.unsqueeze(1).to(device).detach()
            x_dino = x_dino.to(device).detach()
            logits = test_pred(x, model, x_dino if teacher else None)
            h_i = int(h[0])
            w_i = int(w[0])
            logits = logits[0:1, :, :h_i, :w_i]
            y = y[0:1, :, :h_i, :w_i]
            preds = torch.sigmoid(logits)
            iou_metric.update(preds, y)
            niou_metric.update(preds, y)
            fa_pd_metric.update(preds, y)
    _, miou = iou_metric.get()
    _, niou = niou_metric.get()
    fa, pd = fa_pd_metric.get()
    return {"IoU": miou, "nIoU": niou, "FA": fa, "PD": pd}


def update_pseudo_masks(loader, teacher, mask_dir, device):
    for img, name, h, w, img_dino in loader:
        img = img.to(device=device)
        img_dino = img_dino.to(device=device)
        with torch.no_grad():
            output = test_pred(img, teacher, img_dino)
            output = output[:, :, :int(h[0]), :int(w[0])]
            output = torch.sigmoid(output).cpu().data.numpy()
        for i in range(output.shape[0]):
            pred = output[i][0].astype("float32")
            mask_path = os.path.join(mask_dir, name[i])
            prev_label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
            updated = update_gt_update_degen_corr(
                pred,
                prev_label,
                0.5,
                0.5,
                [int(h[i]), int(w[i])],
                degen=0.97,
            )
            cv2.imwrite(mask_path, updated * 255)


def predict_unselected_masks(loader, teacher, mask_dir, device):
    for img, name, h, w, img_dino in loader:
        img = img.to(device=device)
        img_dino = img_dino.to(device=device)
        with torch.no_grad():
            output = test_pred(img, teacher, img_dino)
            output = output[:, :, :int(h[0]), :int(w[0])]
            output = torch.sigmoid(output).cpu().data.numpy()
        for i in range(output.shape[0]):
            pred = output[i][0].astype("float32")
            pred = cv2.resize(pred, (int(w[i]), int(h[i])))
            pred = np.where(pred > 0.5, 255, 0).astype(np.uint8)
            cv2.imwrite(os.path.join(mask_dir, name[i]), pred)


def train_one_epoch(epoch, lr_inner, train_loader, val_loader, teacher, student, alpha, optim_inner, architect, cluster_map, device):
    student.train()
    teacher.train()
    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    train_losses = []
    for data, targets, _, image_dino, img_id in train_loader:
        data = data.to(device)
        targets = targets.unsqueeze(1).to(device)
        image_dino = image_dino.to(device)

        optim_inner.zero_grad()
        loss_inner, logits_t = inner_loss(
            student,
            teacher,
            alpha,
            data,
            targets,
            image_dino,
            img_id,
            cluster_map,
            device,
        )
        loss_inner.backward()
        optim_inner.step()
        train_losses.append(float(loss_inner.item()))

        predictions = torch.sigmoid(logits_t[0] if isinstance(logits_t, (list, tuple)) else logits_t)
        iou_metric.update(predictions, targets)
        niou_metric.update(predictions, targets)

    if epoch % 5 == 0 and len(val_loader) > 0 and len(train_loader) > 0:
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        for _ in range(4):
            try:
                batch_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_train = next(train_iter)
            try:
                batch_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch_val = next(val_iter)
            optim_inner.zero_grad()
            architect.step(
                lamda=None,
                latency=None,
                input_train=batch_train,
                target_train=None,
                input_valid=batch_val,
                target_valid=None,
                eta=lr_inner,
                network_optimizer=optim_inner,
                unrolled=True,
            )

    _, train_iou = iou_metric.get()
    return train_iou, float(np.mean(train_losses)) if train_losses else 0.0


def update_masks(epoch, epochs, num_workers, paths, teacher, device, eval_transform):
    if epoch <= 60 or epoch % 5 != 0:
        return

    teacher.eval()
    print("Updating pseudo labels...")
    update_pseudo_masks(
        DataLoader(
            LabelUpdateDataset(paths["train_img"], eval_transform),
            batch_size=1,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        ),
        teacher,
        paths["train_mask"],
        device,
    )
    update_pseudo_masks(
        DataLoader(
            LabelUpdateDataset(paths["val_img"], eval_transform),
            batch_size=1,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        ),
        teacher,
        paths["val_mask"],
        device,
    )

    if epoch <= int(epochs * 0.8):
        lose_ratio = 0.2 + (epoch - 59) / (0.8 * epochs - 60) * 0.8
        print("Mining hard samples from unselected train samples...")
        predict_unselected_masks(
            DataLoader(
                LabelUpdateDataset(paths["train_unselected_img"], eval_transform),
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
            ),
            teacher,
            paths["train_unselected_mask"],
            device,
        )
        selected_names = select_hard_samples(
            paths["train_unselected_img"],
            paths["train_unselected_mask"],
            paths["train_unselected_points"],
            lose_point_ratio=lose_ratio,
            alarm_point_ratio=5.0,
        )
        print(selected_names)
        print(len(selected_names))
        refine_generated_masks(paths["train_unselected_mask"], paths["train_unselected_points"], selected_names)
        move_hard_samples(
            paths["train_unselected_img"],
            paths["train_unselected_mask"],
            paths["train_unselected_points"],
            paths["train_img"],
            paths["train_mask"],
            paths["train_points"],
            selected_names,
        )
        print("Hard sample transfer finished.")


def train():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = f"{args.model}Net"
    tag = f"_{model_tag}_SIRST3_{args.save_tag}_{timestamp}" if args.save_tag else f"_{model_tag}_SIRST3_{timestamp}"
    print(tag)

    paths = build_paths(args, tag)
    train_sample_names = load_sample_names(paths["train_split"])
    val_sample_names = load_sample_names(paths["val_split"])
    test_sample_names = load_sample_names(paths["test_split"])

    initialize_point_labels(
        paths["train_source_img"],
        paths["train_source_point"],
        paths["train_img"],
        paths["train_mask"],
        paths["train_points"],
        paths["train_unselected_img"],
        paths["train_unselected_mask"],
        paths["train_unselected_points"],
        crop_size=10,
        select_all=False,
        sample_names=train_sample_names,
    )
    initialize_point_labels(
        paths["val_source_img"],
        paths["val_source_point"],
        paths["val_img"],
        paths["val_mask"],
        paths["val_points"],
        crop_size=10,
        select_all=True,
        sample_names=val_sample_names,
    )

    cluster_map, num_clusters = build_cluster_map_from_csv(
        os.path.join(os.path.dirname(__file__), "utils", "ir_image_metrics_with_clusters.csv")
    )
    cal_mean, cal_std = calculate_mean_std(
        [
            (paths["train_source_img"], train_sample_names),
            (paths["val_source_img"], val_sample_names),
        ]
    )
    if args.model == "MLCL":
        student = MLCLNet(in_channels=3)
    elif args.model == "ALCL":
        student = ALCLNet(in_channels=3)
    teacher = DINOCNNTeacher(
        dino_backbone=FrozenDINOBackbone("facebook/dinov3-vits16plus-pretrain-lvd1689m", device=device),
        base_model=student,
        stage_channels=list(student.stage_channels),
        hidden_dim=128,
        residual_gamma=True,
    )
    teacher.to(device)
    student.to(device)
    alpha = torch.nn.Parameter(torch.zeros(num_clusters, device=device))
    optim_inner = torch.optim.AdamW(
        list(student.parameters()) + [p for p in teacher.modulator.parameters() if p.requires_grad],
        lr=args.lr_inner,
    )
    arch_model = ArchModelGN(student, teacher, alpha, cluster_map, device)
    class _ArgsForArchitect:
        arch_learning_rate = args.lr_outer
        arch_learning_rate_alpha = args.lr_alpha
        arch_weight_decay = 1e-2
    architect = Architect(arch_model, _ArgsForArchitect)
    train_transform, eval_transform = build_transforms(mean=cal_mean, std=cal_std)
    best_student = {"IoU": 0.0, "epoch": 0}
    best_teacher = {"IoU": 0.0, "epoch": 0}
    ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print("Epoch:", epoch)

        update_masks(epoch, args.epochs, args.num_workers, paths, teacher, device, eval_transform)
        train_loader = DataLoader(
            TrainPoolDataset(paths["train_img"], paths["train_mask"], 256, train_transform),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            TrainPoolDataset(paths["val_img"], paths["val_mask"], 256, train_transform),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            EvalDataset(
                paths["test_img"],
                paths["test_label"],
                eval_transform,
                names=test_sample_names,
            ),
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        train_miou, train_loss = train_one_epoch(
            epoch,
            args.lr_inner,
            train_loader,
            val_loader,
            teacher,
            student,
            alpha,
            optim_inner,
            architect,
            cluster_map,
            device,
        )

        print("=" * 60)
        if epoch + 1 > args.eval_start_epoch:
            student_metrics = evaluate(test_loader, student, device, teacher=False)
            teacher_metrics = evaluate(test_loader, teacher, device, teacher=True)
            if best_student["IoU"] < student_metrics["IoU"]:
                best_student = {
                    "IoU": student_metrics["IoU"],
                    "epoch": epoch,
                }
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": student.state_dict(),
                        "best_mIoU": best_student["IoU"],
                        "metrics": student_metrics,
                    },
                    os.path.join(ckpt_dir, f"best_student_mIoU_{tag}.pth"),
                )
            if best_teacher["IoU"] < teacher_metrics["IoU"]:
                best_teacher = {
                    "IoU": teacher_metrics["IoU"],
                    "epoch": epoch,
                }
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": teacher.state_dict(),
                        "best_mIoU": best_teacher["IoU"],
                        "metrics": teacher_metrics,
                    },
                    os.path.join(ckpt_dir, f"best_teacher_mIoU_{tag}.pth"),
                )
            print(
                f"Epoch:{epoch + 1} Train IoU:{round(train_miou, 4)} Train Loss:{train_loss:.4f}\n"
                f"Student - IoU:{student_metrics['IoU']:.4f} nIoU:{student_metrics['nIoU']:.4f} PD:{student_metrics['PD']:.4f} FA:{student_metrics['FA']:.8f}\n"
                f"Teacher - IoU:{teacher_metrics['IoU']:.4f} nIoU:{teacher_metrics['nIoU']:.4f} PD:{teacher_metrics['PD']:.4f} FA:{teacher_metrics['FA']:.8f}\n"
                f"Best Student - IoU:{best_student['IoU']:.4f}\n"
                f"Best Teacher - IoU:{best_teacher['IoU']:.4f}"
            )
        else:
            print(f"Epoch:{epoch + 1} Train IoU:{round(train_miou, 4)} Train Loss:{train_loss:.4f}\n")

    print(f"Finished training with tag {tag}")


if __name__ == "__main__":
    train()
