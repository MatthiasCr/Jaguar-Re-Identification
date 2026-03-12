import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
import timm.data


DEFAULT_MEAN = (0.481, 0.457, 0.408)
DEFAULT_STD = (0.268, 0.261, 0.275)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_train_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "train.csv")


def load_test_pairs_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "test.csv")


def encode_labels(df: pd.DataFrame, label_col: str = "ground_truth", encoded_col: str = "label_encoded"):
    label_encoder = LabelEncoder()
    df = df.copy()
    df[encoded_col] = label_encoder.fit_transform(df[label_col])
    return df, label_encoder


def train_val_split(df: pd.DataFrame, val_split: float, seed: int, stratify_col: str = "ground_truth"):
    return train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=df[stratify_col],
    )


def build_eval_frames_from_config(checkpoint_config, label_col: str = "ground_truth"):
    data_dir = Path(checkpoint_config["data_dir"])
    train_df = load_train_df(data_dir)
    train_df, _ = encode_labels(train_df, label_col=label_col)
    _, val_df = train_val_split(
        train_df,
        val_split=float(checkpoint_config["val_split"]),
        seed=int(checkpoint_config["seed"]),
        stratify_col=label_col,
    )

    pairs_df = load_test_pairs_df(data_dir)
    unique_images = sorted(set(pairs_df["query_image"].unique()) | set(pairs_df["gallery_image"].unique()))
    test_df = pd.DataFrame({"filename": unique_images})

    return {
        "data_dir": data_dir,
        "val_df": val_df,
        "pairs_df": pairs_df,
        "test_df": test_df,
    }


def build_transforms_baseline(
    input_size: int,
    train: bool,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    augment: bool = False,
):
    if train and augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (input_size, input_size),
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.03,
                ),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                    ],
                    p=0.15,
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.25),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_transforms(model, input_size: int):
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # extract the normalization parameters from the backbone transform
    normalize_transform = None
    for t in transform.transforms:
        if isinstance(t, transforms.Normalize):
            normalize_transform = t
            break

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize_transform,
        ]
    )


def get_model_normalization(model):
    data_config = timm.data.resolve_model_data_config(model)
    return tuple(data_config["mean"]), tuple(data_config["std"])


def get_crop_position(resized_size: int, crop_size: int, anchor: str):
    max_offset = resized_size - crop_size
    if anchor == "center":
        offset = max_offset // 2
        return offset, offset
    if anchor == "top_left":
        return 0, 0
    if anchor == "top_right":
        return 0, max_offset
    if anchor == "bottom_left":
        return max_offset, 0
    if anchor == "bottom_right":
        return max_offset, max_offset
    raise ValueError(f"Unsupported anchor: {anchor}")


class DeterministicTTATransform:
    def __init__(self, input_size, mean, std, crop_scale=1.0, anchor="center"):
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.crop_scale = crop_scale
        self.anchor = anchor

    def __call__(self, img):
        img = TF.resize(
            img,
            [self.input_size, self.input_size],
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        if self.crop_scale < 1.0:
            crop_size = max(int(round(self.input_size * self.crop_scale)), 1)
            top, left = get_crop_position(self.input_size, crop_size, self.anchor)
            img = TF.crop(img, top, left, crop_size, crop_size)
            img = TF.resize(
                img,
                [self.input_size, self.input_size],
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)
        return img


def build_tta_transform(model, input_size: int, crop_scale: float = 1.0, anchor: str = "center"):
    mean, std = get_model_normalization(model)
    return DeterministicTTATransform(
        input_size=input_size,
        mean=mean,
        std=std,
        crop_scale=crop_scale,
        anchor=anchor,
    )


class JaguarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, label_col: str = "label_encoded", is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.label_col = label_col
        self.is_test = is_test

        if not self.is_test and self.label_col not in self.df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in dataframe")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        img_path = self.img_dir / img_name
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (1, 1))

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, img_name

        return img, torch.tensor(row[self.label_col], dtype=torch.long)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def create_eval_loader(
    df,
    img_dir,
    transform,
    batch_size,
    num_workers=2,
    label_col: str = "label_encoded",
    is_test: bool = False,
    seed: int | None = None,
):
    generator = torch.Generator()
    if seed is None:
        seed = int(os.getenv("PYTHONHASHSEED", "0"))
    generator.manual_seed(seed)

    return DataLoader(
        JaguarDataset(df, img_dir, transform=transform, label_col=label_col, is_test=is_test),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )


def create_dataloaders(
    train_df,
    val_df,
    img_dir,
    input_size,
    batch_size,
    num_workers=2,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    weighted_sampling: bool = False,
    label_col: str = "label_encoded",
    augment: bool = False,
):
    train_transform = build_transforms_baseline(
        input_size=input_size,
        train=True,
        mean=mean,
        std=std,
        augment=augment,
    )
    val_transform = build_transforms_baseline(input_size=input_size, train=False, mean=mean, std=std)

    generator = torch.Generator()
    generator.manual_seed(int(os.getenv("PYTHONHASHSEED", "0")))

    train_dataset = JaguarDataset(train_df, img_dir, train_transform, label_col=label_col)

    if weighted_sampling:
        class_counts = train_df[label_col].value_counts().sort_index()
        class_weights = 1.0 / class_counts
        sample_weights = train_df[label_col].map(class_weights).to_numpy(dtype="float64")
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            pin_memory=False,
        )

    val_loader = DataLoader(
        JaguarDataset(val_df, img_dir, val_transform),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_backbone_dataloaders(
    model,
    train_df,
    val_df,
    img_dir,
    input_size,
    batch_size,
    num_workers=2,
):
    transform = build_transforms(model, input_size=input_size)

    generator = torch.Generator()
    generator.manual_seed(int(os.getenv("PYTHONHASHSEED", "0")))

    train_loader = DataLoader(
        JaguarDataset(train_df, img_dir, transform),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )

    val_loader = DataLoader(
        JaguarDataset(val_df, img_dir, transform),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_test_loader(
    model,
    test_df,
    img_dir,
    input_size,
    batch_size,
    num_workers=2,
):
    test_transform = build_transforms(model, input_size=input_size)

    test_loader = DataLoader(
        JaguarDataset(test_df, img_dir, test_transform, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )
    return test_loader


def create_embedding_dataloaders(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    batch_size,
    num_workers=0,
    weighted_sampling: bool = False,
):
    
    generator = torch.Generator()
    generator.manual_seed(int(os.getenv("PYTHONHASHSEED", "0")))

    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    train_labels_tensor = train_dataset.labels

    if weighted_sampling:
        class_counts = torch.bincount(train_labels_tensor).float()

        # class_weights = 1.0 / class_counts.clamp_min(1.0)
        class_weights = 1.0 / torch.sqrt(class_counts.clamp_min(1.0))


        sample_weights = class_weights[train_labels_tensor].double()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=seed_worker,
            pin_memory=False,
        )

    val_loader = DataLoader(
        EmbeddingDataset(val_embeddings, val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )

    return train_loader, val_loader
