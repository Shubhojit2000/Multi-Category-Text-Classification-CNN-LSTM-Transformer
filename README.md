# Structure of Files and Usage

## `train.py`
This script trains a ResNet model on the training dataset. 

**Arguments:**
- Path to the folder containing training images.
- Path to the directory where the trained model checkpoint will be saved. The model checkpoint will be saved as `resnet_model.pth`.

**Example:**
```bash
python train.py <train_data_dir> <model_ckpt_dir>

## `evaluate.py`
This script evaluates the trained model on a test dataset.

**Arguments:**
- Path to the model checkpoint (resnet_model.pth).
- Path to the folder containing test images.

**Expected Outputs:**
- submission.csv: A CSV file with two columns — image_name and label.
- seg_maps/: A directory containing predicted segmentation maps for each test image.

**Example:**
```bash
python evaluate.py <model_ckpt_path> <test_imgs_dir>
