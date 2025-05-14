# Structure of Files and Usage

## `train.py`
This script trains a ResNet model on the training dataset. 

**Arguments:**
- Path to the folder containing training images.
- Path to the directory where the trained model checkpoint will be saved. The model checkpoint will be saved as `resnet_model.pth`.

**Example:**
```bash
python train.py <train_data_dir> <model_ckpt_dir>
