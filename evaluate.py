import os
import sys
import csv
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#############################
# 1) Model and Helper Functions
#############################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, n, num_classes):
        """
        n: number of residual blocks per group.
        num_classes: number of output classes.
        """
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(num_blocks=n, in_channels=32, 
                                       out_channels=32, stride=1)
        self.layer2 = self._make_layer(num_blocks=n, in_channels=32, 
                                       out_channels=64, stride=2)
        self.layer3 = self._make_layer(num_blocks=n, in_channels=64, 
                                       out_channels=128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, num_blocks, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def compute_multi_layer_saliency_map(model, x, class_idx=None, 
                                     selected_layers=["layer1", "layer2", "layer3"],
                                     layer_weights=None):
    """
    Compute an aggregated saliency map from multiple layers using a Grad-CAM style method.
    """
    model.eval()
    activations = {}
    gradients = {}
    
    # Hook functions to store activations and gradients.
    def forward_hook(module, input, output, layer_name):
        activations[layer_name] = output.detach()
    
    def backward_hook(module, grad_input, grad_output, layer_name):
        gradients[layer_name] = grad_output[0].detach()
    
    hooks = []
    for layer_name in selected_layers:
        layer = getattr(model, layer_name)
        h1 = layer.register_forward_hook(lambda m, i, o, ln=layer_name: forward_hook(m, i, o, ln))
        h2 = layer.register_full_backward_hook(lambda m, gi, go, ln=layer_name: backward_hook(m, gi, go, ln))
        hooks.extend([h1, h2])
    
    # Forward pass.
    x.requires_grad_()
    scores = model(x)
    if class_idx is None:
        class_idx = scores.argmax(dim=1)
    # For a single image, choose the first sample.
    score = scores[0, class_idx if isinstance(class_idx, int) else class_idx[0]]
    
    model.zero_grad()
    score.backward()
    
    for h in hooks:
        h.remove()
    
    if layer_weights is None:
        layer_weights = [1.0 for _ in selected_layers]
    else:
        assert len(layer_weights) == len(selected_layers), "Weights must match number of selected layers."
    
    total_weight = sum(layer_weights)
    aggregated_map = None
    for idx, layer_name in enumerate(selected_layers):
        act = activations[layer_name]  # shape: [1, C, H, W]
        grad = gradients[layer_name]   # shape: [1, C, H, W]
        weights = torch.clamp(grad, min=0)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        cam = torch.nn.functional.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        if aggregated_map is None:
            aggregated_map = layer_weights[idx] * cam_np
        else:
            aggregated_map += layer_weights[idx] * cam_np

    aggregated_map = aggregated_map / total_weight
    aggregated_map = (aggregated_map - aggregated_map.min()) / (aggregated_map.max() - aggregated_map.min() + 1e-8)
    aggregated_map = cv2.GaussianBlur(aggregated_map.astype(np.float32), (23, 23), 4)
    
    return aggregated_map

def graphcut_segmentation(img, saliency):
    """
    Given an image and its aggregated saliency map, compute the object segmentation mask using GrabCut.
    """
    fg_thresh = np.quantile(saliency, 0.85)
    bg_thresh = np.quantile(saliency, 0.15)
    
    grabcut_mask = np.full(saliency.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[saliency >= fg_thresh] = cv2.GC_FGD
    grabcut_mask[saliency <= bg_thresh] = cv2.GC_BGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img, grabcut_mask, None, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_MASK)
    output_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_mask, connectivity=8)
    if num_labels > 1:
        largest_cc = 0
        max_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                largest_cc = i
        final_mask = (labels == largest_cc).astype(np.uint8)
    else:
        final_mask = output_mask
    
    return final_mask

def post_process_mask(mask, kernel_size=5, iterations=1):
    """
    Refine the segmentation mask using morphological operations.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened

#############################
# 2) Custom Dataset for Test Images
#############################

class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # List only image files (adjust extensions as needed)
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = img
        return img_tensor, img_name, img_path

#############################
# 3) Main Evaluation Routine
#############################

def main():
    # Expected usage: python evaluate.py <model_ckpt_path> <test_imgs_dir>
    model_ckpt_path = sys.argv[1]
    test_imgs_dir = sys.argv[2]
    
    # Assume the mapping JSON file is saved alongside the model checkpoint.
    mapping_path = "class_mapping.json"
    with open(mapping_path, "r") as f:
        idx_to_label = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hyperparameters: adjust 'n' and 'num_classes' as needed.
    n = 2  
    num_classes = 100
    
    # Initialize the model and load checkpoint.
    model = CustomResNet(n=n, num_classes=num_classes)
    state_dict = torch.load(model_ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Define test transforms (should match training transforms).
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225)),
    ])
    
    # Create the test dataset and loader.
    test_dataset = TestImageDataset(img_dir=test_imgs_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Folder to save segmentation maps.
    seg_maps_dir = os.path.join(os.getcwd(), "seg_maps")
    os.makedirs(seg_maps_dir, exist_ok=True)
    
    submission_rows = []  # List to store CSV rows
    # Define layer weights as desired.
    layer_weights = [0.1, 0.2, 0.7]
    
    for img_tensor, img_name, img_path in test_loader:
        # Move image tensor to device.
        img_tensor = img_tensor.to(device)
        
        # Forward pass for classification.
        output = model(img_tensor)
        _, pred_idx_tensor = output.max(1)
        pred_idx = pred_idx_tensor.item()
        # Retrieve label using the mapping; note JSON keys are strings.
        pred_label = idx_to_label.get(str(pred_idx), "Unknown")
        
        # Compute saliency map.
        saliency = compute_multi_layer_saliency_map(model, img_tensor, class_idx=pred_idx,
                                                     selected_layers=["layer1", "layer2", "layer3"],
                                                     layer_weights=layer_weights)
        # Read original image using cv2.
        orig_img = cv2.imread(img_path[0])
        if orig_img is None:
            print(f"Warning: Could not read image: {img_path[0]}")
            continue
        h, w = orig_img.shape[:2]
        
        # Upsample saliency map to original image dimensions.
        saliency_full = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Obtain segmentation mask directly on the original image.
        pred_mask = graphcut_segmentation(orig_img, saliency_full)
        pred_mask = post_process_mask(pred_mask)
        
        # Save segmentation map in the seg_maps folder.
        seg_map_path = os.path.join(seg_maps_dir, img_name[0])
        cv2.imwrite(seg_map_path, (pred_mask * 255).astype(np.uint8))
        
        # Append result for CSV.
        submission_rows.append([img_name[0], pred_label])
    
    # Write submission CSV.
    csv_filename = "submission.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "label"])
        writer.writerows(submission_rows)
    
    # print(f"Submission CSV saved as: {csv_filename}")
    # print(f"Segmentation maps saved in folder: {seg_maps_dir}")

if __name__ == "__main__":
    main()