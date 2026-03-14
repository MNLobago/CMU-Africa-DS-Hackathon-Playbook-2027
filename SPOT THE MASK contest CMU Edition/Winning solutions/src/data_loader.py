import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class MaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = int(row["target"]) if "target" in row else -1
        return image, target, row["image"]


class MaskDataLoader:
    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, images_dir: str, train_csv: str,
                 sample_submission_csv: str, val_split: float = 0.15,
                 batch_size: int = 32, random_state: int = 42):
        self.images_dir = images_dir
        self.batch_size = batch_size

        train_df = pd.read_csv(train_csv)

        train_df.columns = [c.strip().lower() for c in train_df.columns]
        if "id" in train_df.columns and "image" not in train_df.columns:
            train_df.rename(columns={"id": "image"}, inplace=True)

        submission_df = pd.read_csv(sample_submission_csv)
        submission_df.columns = [c.strip().lower() for c in submission_df.columns]
        if "id" in submission_df.columns and "image" not in submission_df.columns:
            submission_df.rename(columns={"id": "image"}, inplace=True)

        train_ids = set(train_df["image"].values)
        self.test_df = submission_df[["image"]].copy()

        tr, val = train_test_split(
            train_df, test_size=val_split,
            stratify=train_df["target"], random_state=random_state
        )
        self.train_df = tr.reset_index(drop=True)
        self.val_df = val.reset_index(drop=True)

    def get_train_loader(self) -> DataLoader:
        ds = MaskDataset(self.train_df, self.images_dir, self.TRAIN_TRANSFORMS)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)

    def get_val_loader(self) -> DataLoader:
        ds = MaskDataset(self.val_df, self.images_dir, self.VAL_TRANSFORMS)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    def get_test_loader(self) -> DataLoader:
        ds = MaskDataset(self.test_df, self.images_dir, self.VAL_TRANSFORMS)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)