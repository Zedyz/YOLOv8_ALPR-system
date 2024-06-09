## Requirements
If needed, install the requirements file using a conda env.

## Model and Dataset Downloads
You can download all necessary models and datasets via the links provided below:

- **Models**: [Download Models](https://drive.google.com/drive/folders/1hP9Q7bW9zBOvUIfMmZ-XrLuQhWhB2qqc?usp=sharing)
  - `model_1`: Trained for license plate detection.
  - `model_2`: Trained for character detection.
  - `model_3`: Custom U-Net trained for license plate denoising.
- **Datasets**: While not required for running the code, datasets can be downloaded if needed.
  - [Rodosol ALPR Dataset](https://github.com/raysonlaroca/rodosol-alpr-dataset/)
  - [Vehicle Rear Dataset](https://github.com/icarofua/vehicle-rear?tab=readme-ov-file)

## Model Setup
Follow these steps to set up each model correctly:

1. **License Plate Detection Model (Model 1)**
   - Download `model_1_trained_for_license_plate_detection` from the Google Drive `model_1` folder.
   - Place its contents in the `./all_models/model_1` folder within your project directory.

2. **Character Detection Model (Model 2)**
   - Download `model_2_trained_for_character_detection` from the Google Drive `model_2` folder.
   - Place its contents in the `./all_models/model_2` folder within your project directory.

3. **License Plate Denoising Model (Model 3)**
   - Download `denoising_checkpoint.pth` from the Google Drive `model_3` folder.
   - Place it in the `./all_models/model_3_custom_UNet_trained_for_license_plate_denoising` folder within your project directory.

## Description of each project folder

- all_datasets
  - Contains all dataset required for model training and testing

- all_models
  - Contains all folders for the checkpoints and weights for the trained models

- experiments
  - Contains the results from each experiment run

- illustrations
  - Contains illustrations used in the report and project presentation

- utilities
  - contains code for generating datasets, image enhancing techniques, training methods and methods used for experiments.