📁 Structure of Files and Usage
<details> <summary><strong>train.py</strong></summary>
This script trains a ResNet model on the training dataset.

Arguments:

Path to the folder containing training images.

Path to the directory where the trained model checkpoint will be saved.

The model checkpoint will be saved as resnet_model.pth.

Example:

bash
Copy
Edit
python train.py <train_data_dir> <model_ckpt_dir>
</details> <details> <summary><strong>evaluate.py</strong></summary>
This script evaluates the trained model on a test dataset.

Arguments:

Path to the model checkpoint (resnet_model.pth)

Path to the folder containing test images

Expected Outputs:

submission.csv: A CSV file with two columns — image_name and label.

seg_maps/: A directory containing predicted segmentation maps for each test image.

Example:

bash
Copy
Edit
python evaluate.py <model_ckpt_path> <test_imgs_dir>
</details>
📦 Dataset
You can access the dataset here

Replace <INSERT_DATASET_LINK_HERE> with the actual dataset link.

🧠 Model
Base Architecture: ResNet

Task: Classification and Segmentation of Butterfly and Moth species (100 classes)

📌 Repository Overview
File	Description
train.py	Trains a ResNet model on the dataset
evaluate.py	Evaluates model and saves CSV + segmentation outputs
