import os

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk


class ImageNetSmall(Dataset):

    def __init__(
        self,
        is_val=False,
        val_rate=0.005,
        seed=1437,
        cache_dir="D:/AI/HuggingFace/datasets",
        transforms=None,
        need_transform=True,
        disk_dir=None,
    ):
        if disk_dir is not None:
            disk_path = os.path.join(disk_dir, "image_net_small_dataset")
            if not os.path.exists(disk_dir):
                os.makedirs(disk_dir)
            if os.path.exists(disk_path):
                self.dataset = load_from_disk(
                    disk_path, keep_in_memory=True
                ).with_format("torch")
                return
        self.dataset = load_dataset("israfelsr/mm_tiny_imagenet", cache_dir=cache_dir)
        self.split_dataset = self.dataset["train"].train_test_split(
            test_size=val_rate, seed=seed, shuffle=True
        )
        if is_val:
            self.dataset = self.split_dataset["test"]
        else:
            self.dataset = self.split_dataset["train"]
        if need_transform:
            if transforms is None:
                from torchvision.transforms import (
                    Compose,
                    ToTensor,
                    Resize,
                    Normalize,
                    ColorJitter,
                )

                transforms = Compose(
                    [
                        Resize((224, 224)),
                        ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                        ),
                        ToTensor(),
                        # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                )

            def _transforms(examples):
                examples["image"] = [
                    transforms(image.convert("RGB")) for image in examples["image"]
                ]
                return examples

            if disk_dir is None:
                self.dataset.set_transform(_transforms)
            else:
                self.dataset = self.dataset.map(
                    _transforms, batched=True, batch_size=64
                )
        if disk_dir is not None:
            self.dataset.save_to_disk(disk_path)
            self.dataset = load_from_disk(disk_path).with_format("torch")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def collate_fn(self, batchs):
        images = []
        labels = []
        for b in batchs:
            images.append(b["image"])
            labels.append(b["label"])
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels
