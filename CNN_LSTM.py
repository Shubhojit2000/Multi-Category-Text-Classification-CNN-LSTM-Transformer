# CNN + LSTM
import pandas as pd
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.manual_seed(42)

# Define regex-based tokenizer
def tokenizer(text):
    return re.findall(r'\w+', text.lower())

# Load data
train_df = pd.read_csv('TrainData.csv')
test_df = pd.read_csv('TestLabels.csv')

# Build vocabulary using all training texts
counter = Counter()
train_texts_all = train_df['Text'].astype(str).tolist()
for text in train_texts_all:
    counter.update(tokenizer(text))
vocab = {'<pad>': 0, '<unk>': 1}
for word, count in counter.items():
    if count >= 2:  # Minimum frequency threshold
        vocab[word] = len(vocab)

# Dataset parameters
max_len = 400  # Adjust based on data analysis
label_map = {'business': 0, 'tech': 1, 'politics': 2, 'sport': 3, 'entertainment': 4}

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = [label_map[label] for label in labels]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = tokenizer(text)
        numericalized = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        if len(numericalized) > self.max_len:
            numericalized = numericalized[:self.max_len]
        else:
            numericalized += [self.vocab['<pad>']] * (self.max_len - len(numericalized))
        return torch.tensor(numericalized), torch.tensor(label)

# Split the training data into train (80%) and validation (20%)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['Text'].tolist(), train_df['Category'].tolist(),
    test_size=0.2, random_state=42
)

# Create datasets and dataloaders for training, validation, and testing
batch_size = 64
train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
val_dataset = TextDataset(val_texts, val_labels, vocab, max_len)
test_dataset = TextDataset(
    test_df['Text'].tolist(), 
    test_df['Label - (business, tech, politics, sport, entertainment)'].tolist(), 
    vocab, max_len
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_filters, filter_sizes, dropout_rate):
        super(CNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 + num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # LSTM branch
        lstm_out, _ = self.lstm(embedded)
        lstm_pooled, _ = torch.max(lstm_out, dim=1)  # Global max pooling
        
        # CNN branch
        embedded_permuted = embedded.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        conved = [F.relu(conv(embedded_permuted)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cnn_pooled = torch.cat(pooled, dim=1)
        
        # Concatenate and classify
        combined = torch.cat([lstm_pooled, cnn_pooled], dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        return output

# Initialize model and training components
vocab_size = len(vocab)
embed_dim = 300
hidden_dim = 256
num_classes = 5
num_filters = 150
filter_sizes = [2, 3, 4, 5]
dropout_rate = 0.5
num_epochs = 15

model = CNNLSTM(vocab_size, embed_dim, hidden_dim, num_classes, num_filters, filter_sizes, dropout_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

# For tracking metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_accuracy = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    
    # Evaluate on validation set
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'    Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
    print(f'    Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_val_acc:.4f}')
    
    scheduler.step(epoch_val_acc)
    # Save best model based on validation accuracy
    if epoch_val_acc > best_val_accuracy:
        best_val_accuracy = epoch_val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# Plot training and validation loss and accuracy per epoch
epochs = range(1, num_epochs+1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('loss_and_accuracy_per_epoch.png')

# Load the best model (based on validation accuracy)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Evaluate on the test set
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute test metrics
test_accuracy = accuracy_score(all_labels, all_preds)
test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print("\nTest Metrics:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")