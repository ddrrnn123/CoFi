import os
import torch
import numpy as np
import torch.nn as nn
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
from albumentations.pytorch import ToTensorV2


class DeepLabV3Wrapper(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_pretrained=True):

        super().__init__()

        # load ResNet50‐backboned DeepLabV3
        weights = (
            DeepLabV3_ResNet50_Weights.DEFAULT if use_pretrained else None
        )
        self.deeplab = deeplabv3_resnet50(weights=weights)
        if in_channels != 3:
            self.deeplab.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace the classifier head for our # of outputs
        self.deeplab.classifier[4] = nn.Conv2d(
            self.deeplab.classifier[4].in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        out = self.deeplab(x)["out"]
        return out

class SupportDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(512,512)):

        # gather and sort only valid image filenames
        valid_ext = (".png", ".jpg", ".jpeg", ".tif")
        self.img_files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(valid_ext)
        ])
        self.mask_files = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith(valid_ext)
        ])
        if len(self.img_files) != len(self.mask_files):
            raise ValueError(f"Found {len(self.img_files)} images but {len(self.mask_files)} masks")

        h, w = size

        # Albumentations pipeline applies same transforms to image & mask
        self.aug = A.Compose([
            A.Resize(height=h, width=w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.5),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        
        img_np  = np.array(Image.open(self.img_files[idx]).convert("L"))
        mask_np = np.array(Image.open(self.mask_files[idx]).convert("L"))

        # apply Albumentations
        augmented = self.aug(image=img_np, mask=mask_np)
        img_t  = augmented["image"]               # [1, H, W], already normalized
        mask_t = augmented["mask"][None, ...]     # [1, H, W], raw 0–255

        # binarize mask and scale to [0,1]
        mask_t = (mask_t > 0).float()
        return img_t, mask_t


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.requires_grad_(False)

def main():
    # Paths and hyperparameters
    SUBSET_IMG_DIR = "/support_images"
    SUBSET_MSK_DIR = "/support_masks"
    MODEL_PATH  = "/checkpoint/trained_deeplabv3checkpoint.pth"
    SIZE        = (512, 512)
    EPOCHS      = 60
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Initialize model and data loader
    model = DeepLabV3Wrapper(in_channels=1, out_channels=1).to(DEVICE)
    train_loader = DataLoader(SupportDataset(SUBSET_IMG_DIR, SUBSET_MSK_DIR, SIZE),
                              batch_size=1, shuffle=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        
        model.train()
        freeze_batchnorm(model)
  
        total_loss = 0.0
        for img, msk in train_loader:
            img, msk = img.to(DEVICE), msk.to(DEVICE)
            logits = model(img)

            loss = 0.7 * bce_loss(logits, msk) + 0.3 * dice_loss(logits, msk)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}/{EPOCHS}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Saved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    main()