import os
import argparse
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description="Train models with custom parameters")

parser.add_argument('--data', type=str, default=None)
parser.add_argument('--image_size', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--model_name', type=str, default="run_x")
parser.add_argument('--resume', type=str, default=True)
parser.add_argument('--model_type', type=str, default="character", choices=['license_plates', 'character'])

args = parser.parse_args()
model_path = '../../all_models/untrained_yolov8/yolov8l-obb.pt'
if args.model_type == 'license_plates':
    args.data = 'all_datasets/dataset_1_licenseplates/data.yaml'
elif args.model_type == 'character':
    args.data = 'all_datasets/dataset_2_characters/data.yaml'

model = YOLO(model_path)

# data augmentation settings

if args.model_type == 'character':
    hsv_h, hsv_s, hsv_v = 0, 0, 0
    degrees, translate, scale, shear, perspective = 1.0, 0, 0.1, 0, 0.001
    flipud, fliplr, mosaic, mixup = 0, 0, 0, 0
else:
    hsv_h, hsv_s, hsv_v = 0.015, 0.7, 0.6
    degrees, translate, scale, shear, perspective = 10.0, 0.1, 0.3, 3.0, 0.002
    flipud, fliplr, mosaic, mixup = 0.0, 0.5, 0.5, 0.1

model.train(
    data=args.data,
    imgsz=args.image_size,
    epochs=args.num_epochs,
    name=args.model_name,
    hsv_h=hsv_h,
    hsv_s=hsv_s,
    hsv_v=hsv_v,
    degrees=degrees,
    translate=translate,
    scale=scale,
    shear=shear,
    perspective=perspective,
    flipud=flipud,
    fliplr=fliplr,
    mosaic=mosaic,
    mixup=mixup,
    save=True,
    lr0=1e-3,
    plots=True,
    batch=64
)


