from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AcneDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        class_column: str = "class",
        filename_column: str = "filename",
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        split_dir = self.root_dir / split
        csv_path = split_dir / "_annotations.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # filename, class만 사용 + 같은 이미지가 여러 박스를 가질 수 있으므로 filename 기준 중복 제거
        df = df[[filename_column, class_column]].drop_duplicates(
            subset=[filename_column]
        )

        self.filename_column = filename_column
        self.class_column = class_column
        self.df = df

        # 라벨 인덱스 매핑
        classes = sorted(df[class_column].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # 기본 transform (없으면 ImageNet 통계 기준으로 세팅)
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        print(
            f"[AcneDataset] split={split}, samples={len(self.df)}, classes={classes}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row[self.filename_column]
        label_name = row[self.class_column]

        img_path = self.root_dir / self.split / img_name
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[label_name]

        return image, label
