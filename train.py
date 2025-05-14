import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Define a residual block consisting of two 3x3 convolutional layers.
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
        
        # If the input and output dimensions differ, or if downsampling is needed,
        # use a 1x1 convolution to match dimensions.
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

# Define the custom ResNet architecture.
class CustomResNet(nn.Module):
    def __init__(self, n, num_classes):
        """
        n: number of residual blocks per group (each block has 2 conv layers)
        num_classes: number of output classes
        """
        super(CustomResNet, self).__init__()
        # Initial convolution layer: from 3 input channels to 32 output channels.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Create three groups of residual blocks.
        # Group 1: feature map 224x224, 32 filters.
        self.layer1 = self._make_layer(num_blocks=n, in_channels=32, 
                                       out_channels=32, stride=1)
        # Group 2: feature map 112x112, 64 filters (downsample with stride=2).
        self.layer2 = self._make_layer(num_blocks=n, in_channels=32, 
                                       out_channels=64, stride=2)
        # Group 3: feature map 56x56, 128 filters (downsample with stride=2).
        self.layer3 = self._make_layer(num_blocks=n, in_channels=64, 
                                       out_channels=128, stride=2)
        
        # Global average pooling and a fully connected output layer.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, num_blocks, in_channels, out_channels, stride):
        layers = []
        # The first block in the group may perform downsampling.
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # The remaining blocks keep the same dimensions.
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Pass through the three groups.
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global average pooling and final classification.
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def main():
    # COMMAND LINE PATHS
    train_path = sys.argv[1]
    model_ckpt_dir = sys.argv[2]

    # Hyperparameters and settings.
    class Args:
        n = 2
        num_classes = 100
        epochs = 40
        batch_size = 64
        lr = 0.1
        momentum = 0.9
        weight_decay = 5e-4

    args = Args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    # Data augmentation and normalization for training.
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225)),
    ])
    
    # Create training dataset from the folder structure.
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4)

    # Create validation dataset.
    # val_dataset = datasets.ImageFolder(root="/kaggle/input/ail721data2/Butterfly/Butterfly/valid", transform=transform_train)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
    #                           shuffle=False, num_workers=4)
    
    # Get class index to label mapping.
    class_to_idx = train_dataset.class_to_idx  # {class_name: class_index}
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # {class_index: class_name}

    # Save the mapping to a JSON file.
    mapping_path = "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(idx_to_class, f)
    # print(f"Class index-to-label mapping saved to {mapping_path}")

    # Build the model.
    model = CustomResNet(n=args.n, num_classes=args.num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    # Use CosineAnnealingLR: anneal LR following a cosine schedule over the total number of epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Lists to store training and validation metrics for plotting.
    # train_losses = []
    # train_accuracies = []
    # val_losses = []
    # val_accuracies = []
    
    # Save the model checkpoint as 'resnet_model.pth' in the specified directory.
    os.makedirs(model_ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_ckpt_dir, "resnet_model.pth")
    
    for epoch in range(1, args.epochs + 1):

        model.train()
        # running_loss = 0.0
        # correct = 0
        # total = 0
        # Print current learning rate
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch {epoch}: Learning Rate: {current_lr:.6f}")
    
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Step the cosine annealing scheduler.
        scheduler.step()
        
        torch.save(model.state_dict(), checkpoint_path)
        # print(f"Model checkpoint saved to {checkpoint_path}")
        
    torch.save(model.state_dict(), checkpoint_path)
    # print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Optionally, plot the training and validation loss and accuracy.
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Val Loss')
    # plt.title('Loss vs. Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.plot(train_accuracies, label='Train Accuracy')
    # plt.plot(val_accuracies, label='Val Accuracy')
    # plt.title('Accuracy vs. Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()