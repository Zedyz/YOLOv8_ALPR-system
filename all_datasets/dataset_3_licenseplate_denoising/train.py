import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import torch.nn.functional as F

from all_models.model_3_custom_UNet_trained_for_license_plate_denoising.unet_model import UNet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
batch_size = 4
epochs = 150

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True).eval()
        self.features = nn.Sequential(*list(resnet50.children())[:8])
        for param in self.features.parameters():
            param.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input, target):
        input = input.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        input = self.normalize(input)
        target = self.normalize(target)
        input_features = self.features(input)
        target_features = self.features(target)
        return F.mse_loss(input_features, target_features)

class CombinedLoss(nn.Module):
    def __init__(self, perceptual_loss, weight_perceptual, weight_l1):
        super().__init__()
        self.perceptual_loss = perceptual_loss
        self.l1_loss = nn.L1Loss()
        self.weight_perceptual = weight_perceptual
        self.weight_l1 = weight_l1

    def forward(self, input, target):
        loss_perceptual = self.perceptual_loss(input, target)
        loss_l1 = self.l1_loss(input, target)
        combined_loss = self.weight_perceptual * loss_perceptual + self.weight_l1 * loss_l1
        return combined_loss



class PairedImageDataset(Dataset):
    def __init__(self, root_no_timestamp, root_with_timestamp, transform=None):
        self.root_no_timestamp = root_no_timestamp
        self.root_with_timestamp = root_with_timestamp
        self.transform = transform
        self.no_timestamp_images = [os.path.join(root_no_timestamp, f) for f in os.listdir(root_no_timestamp)]
        self.with_timestamp_images = [os.path.join(root_with_timestamp, f) for f in os.listdir(root_with_timestamp)]

    def __len__(self):
        return min(len(self.no_timestamp_images), len(self.with_timestamp_images))

    def __getitem__(self, idx):
        no_timestamp_path = self.no_timestamp_images[idx]
        with_timestamp_path = self.with_timestamp_images[idx]
        image_no_timestamp = Image.open(no_timestamp_path).convert('RGB')
        image_with_timestamp = Image.open(with_timestamp_path).convert('RGB')
        if self.transform:
            image_no_timestamp = self.transform(image_no_timestamp)
            image_with_timestamp = self.transform(image_with_timestamp)
        return image_with_timestamp, image_no_timestamp

transform = transforms.Compose([
    transforms.Resize((160, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

def main():
    train_dataset = PairedImageDataset("./clean_images/train", "./noisy_images/train", transform=transform)
    val_dataset = PairedImageDataset("./clean_images/validation", "./noisy_images/validation", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda")
    model = UNet().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    combined_loss = CombinedLoss(perceptual_loss, weight_perceptual=0.85, weight_l1=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # load checkpoint if it exists
    checkpoint_path = 'epoch_85.pth'
    start_epoch = 68
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")

    losses_train = []
    losses_val = []

    checkpoint_dir = './'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    loss_log_path = os.path.join(checkpoint_dir, "loss_log.txt")
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w") as loss_log:
            loss_log.write("Epoch,Train Loss,Val Loss,Learning Rate\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss_train = running_loss / len(train_loader)
        losses_train.append(avg_loss_train)

        model.eval()
        running_loss_val = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                val_loss = combined_loss(outputs, labels)
                running_loss_val += val_loss.item()

        avg_loss_val = running_loss_val / len(val_loader)
        losses_val.append(avg_loss_val)

        scheduler.step(avg_loss_val)
        current_lr = scheduler.optimizer.param_groups[0]['lr']

        with open(loss_log_path, "a") as loss_log:
            loss_log.write(f"{epoch + 1},{avg_loss_train:.4f},{avg_loss_val:.4f},{current_lr:.6f}\n")
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_val:.4f}, LR: {current_lr:.6f}")

        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss_train,
            'val_loss': avg_loss_val,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    main()
