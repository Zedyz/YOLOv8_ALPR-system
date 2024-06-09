import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = self.conv_block(1, 64)
        self.down2 = self.conv_block(64, 128, pool=True)
        self.down3 = self.conv_block(128, 256, pool=True)
        self.down4 = self.conv_block(256, 512, pool=True)
        self.down5 = self.conv_block(512, 1024, pool=True)
        self.down6 = self.conv_block(1024, 2048, pool=True)

        # Bottleneck
        self.bottleneck = self.conv_block(2048, 2048)

        # Upsampling path
        self.up5 = self.up_conv_block(2048, 1024)
        self.up4 = self.up_conv_block(1024, 512)
        self.up3 = self.up_conv_block(512, 256)
        self.up2 = self.up_conv_block(256, 128)
        self.up1 = self.up_conv_block(128, 64)

        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        if pool:
            layers.insert(0, nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Downsampling
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        # Bottleneck
        x = self.bottleneck(x6)

        # Upsampling + Skip connections
        x = self.up5(x) + x5
        x = self.up4(x) + x4
        x = self.up3(x) + x3
        x = self.up2(x) + x2
        x = self.up1(x) + x1

        # Final layer
        x = self.final(x)
        return self.sigmoid(x)

    def predict(self, image_path, model_path):
        device = torch.device("cpu")
        model = UNet().to(device)
        checkpoint_path = model_path
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        transform = transforms.Compose([
            transforms.Resize((160, 512)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        input_image = Image.open(image_path).convert('L')
        input_tensor = transform(input_image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_image_pil = transforms.ToPILImage()(output_tensor.squeeze().cpu())
        return output_image_pil
