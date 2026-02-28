import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DEFAULT_MEAN = (0.481, 0.457, 0.408)
DEFAULT_STD = (0.268, 0.261, 0.275)


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


def build_transforms(input_size: int, train: bool, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.25),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
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


def create_dataloaders(
    train_df,
    val_df,
    img_dir,
    input_size,
    batch_size,
    num_workers=2,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
):
    train_transform = build_transforms(input_size=input_size, train=True, mean=mean, std=std)
    val_transform = build_transforms(input_size=input_size, train=False, mean=mean, std=std)

    generator = torch.Generator()
    generator.manual_seed(int(os.getenv("PYTHONHASHSEED", "0")))

    train_loader = DataLoader(
        JaguarDataset(train_df, img_dir, train_transform),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        JaguarDataset(val_df, img_dir, val_transform),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_backbone_dataloaders(
    train_df,
    val_df,
    img_dir,
    input_size,
    batch_size,
    num_workers=2,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
):
    transform = build_transforms(input_size=input_size, train=False, mean=mean, std=std)

    generator = torch.Generator()
    generator.manual_seed(int(os.getenv("PYTHONHASHSEED", "0")))

    train_loader = DataLoader(
        JaguarDataset(train_df, img_dir, transform),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        JaguarDataset(val_df, img_dir, transform),
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_test_loader(
    test_df,
    img_dir,
    input_size,
    batch_size,
    num_workers=2,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
):
    test_transform = build_transforms(input_size=input_size, train=False, mean=mean, std=std)
    test_loader = DataLoader(
        JaguarDataset(test_df, img_dir, test_transform, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
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
):
    
    generator = torch.Generator()
    generator.manual_seed(int(os.getenv("PYTHONHASHSEED", "0")))

    train_loader = DataLoader(
        EmbeddingDataset(train_embeddings, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=False,
    )

    val_loader = DataLoader(
        EmbeddingDataset(val_embeddings, val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
        pin_memory=False,
    )

    return train_loader, val_loader
