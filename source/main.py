import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.data_loader import BrainMRIDataset
from src.model import BrainTumorCNN
from source.train import train_model
from src.evaluate import evaluate_model

# 📁 Root directory pass karo
root_dir = "C:/Users/Admin/Downloads/Tumor/data/dataset"

# 🔄 Transformations for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 📊 Dataset load
dataset = BrainMRIDataset(root_dir, transform=transform)

# 🏆 Train-Test Split (80%-20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 🔁 DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ⚡ Initialize model
model = BrainTumorCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🏋️ Train the model
print("🚀 Training started...")
train_model(model, train_loader, criterion, optimizer, epochs=10)

# 📊 Evaluate the model
print("📈 Evaluating the model...")
evaluate_model(model, test_loader)

print("✅ Done! 🎉")
