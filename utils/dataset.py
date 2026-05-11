import os
import random

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

DINO_TRANSFORM = A.Compose(
    [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


def load_sample_names(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        names = []
        for line in f:
            name = line.strip()
            if name:
                names.append(name if name.endswith(".png") else f"{name}.png")
    return names


def build_paths(args, tag=None):
    if args.dataset_root:
        root = os.path.abspath(os.path.expanduser(args.dataset_root))
        dataset_dir = root if os.path.basename(root) == "SIRST3" else os.path.join(root, "SIRST3")
    else:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "SIRST3"))

    paths = {
        "dataset_dir": dataset_dir,
        "train_source_img": os.path.join(dataset_dir, "images"),
        "train_source_point": os.path.join(dataset_dir, "masks_centroid"),
        "train_split": os.path.join(dataset_dir, "mode", "train.txt"),
        "val_source_img": os.path.join(dataset_dir, "images"),
        "val_source_point": os.path.join(dataset_dir, "masks_centroid"),
        "val_split": os.path.join(dataset_dir, "mode", "val.txt"),
        "test_source_img": os.path.join(dataset_dir, "images"),
        "test_source_label": os.path.join(dataset_dir, "labels"),
        "test_split": os.path.join(dataset_dir, "mode", "test.txt"),
    }

    if tag is None:
        return paths

    pseudo_root = os.path.join(dataset_dir, "pseudo_labels", tag)
    paths.update(
        {
            "test_img": paths["test_source_img"],
            "test_label": paths["test_source_label"],
            "train_img": os.path.join(pseudo_root, "train", "images"),
            "train_mask": os.path.join(pseudo_root, "train", "masks"),
            "train_points": os.path.join(pseudo_root, "train", "points"),
            "val_img": os.path.join(pseudo_root, "val", "images"),
            "val_mask": os.path.join(pseudo_root, "val", "masks"),
            "val_points": os.path.join(pseudo_root, "val", "points"),
            "train_unselected_img": os.path.join(pseudo_root, "train_unselected", "images"),
            "train_unselected_mask": os.path.join(pseudo_root, "train_unselected", "masks"),
            "train_unselected_points": os.path.join(pseudo_root, "train_unselected", "points"),
        }
    )
    return paths


def build_transforms(mean, std):
    train_transform = A.Compose(
        [
            A.SomeOf(
                [
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=0.2),
                    A.Rotate(limit=45, p=0.3),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
                    A.GaussNoise(p=0.2),
                    A.NoOp(),
                    A.NoOp(),
                ],
                3,
                p=0.5,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    eval_transform = A.Compose(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return train_transform, eval_transform


def calculate_mean_std(image_dirs_and_names):
    mean_list = []
    std_list = []
    seen = set()

    for image_dir, names in image_dirs_and_names:
        for name in names:
            key = (image_dir, name)
            if key in seen:
                continue
            seen.add(key)
            image = np.array(Image.open(os.path.join(image_dir, name)).convert("L"), dtype=np.float32)
            mean_list.append(float(image.mean()))
            std_list.append(float(image.std()))

    if not mean_list:
        raise ValueError("No images found while calculating mean/std.")

    return float(np.mean(mean_list) / 255.0), float(np.mean(std_list) / 255.0)


def random_crop(img, mask, patch_size):
    h, w, _ = img.shape
    if min(h, w) < patch_size:
        pad_h = max(h, patch_size) - h
        pad_w = max(w, patch_size) - w
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        h, w, _ = img.shape

    h_start = random.randint(0, h - patch_size)
    w_start = random.randint(0, w - patch_size)
    return img[h_start : h_start + patch_size, w_start : w_start + patch_size], mask[
        h_start : h_start + patch_size, w_start : w_start + patch_size
    ]


def pad_to_multiple(image, mask=None, multiple=32):
    h, w = image.shape[:2]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    if mask is None:
        return image, h, w
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return image, mask, h, w


class TrainPoolDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, name)).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(float)
        image_patch, mask_patch = random_crop(image, mask, self.patch_size)
        transformed = self.transform(image=image_patch, mask=mask_patch)
        return (
            transformed["image"],
            transformed["mask"],
            np.zeros((1, self.patch_size, self.patch_size), dtype=np.int64),
            DINO_TRANSFORM(image=image_patch)["image"],
            os.path.splitext(name)[0],
        )


class EvalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, names):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list(names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, name)).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(float)
        image, mask, h, w = pad_to_multiple(image, mask, multiple=32)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"], h, w, DINO_TRANSFORM(image=image)["image"]


class LabelUpdateDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        image, h, w = pad_to_multiple(image, multiple=32)
        transformed = self.transform(image=image)
        return transformed["image"], name, h, w, DINO_TRANSFORM(image=image)["image"]
