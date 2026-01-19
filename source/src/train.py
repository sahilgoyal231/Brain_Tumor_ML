import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from source.create_dummy_model import BrainTumorCNN  # 🧠 CNN Architecture

# ✅ Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 0.001

# 🔄 Data Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 📁 Dataset Load
dataset = datasets.ImageFolder(root='dataset/', transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 🧠 Model Init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainTumorCNN().to(device)

# ⚙️ Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 🚀 Training Loop
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"📅 Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 💾 Save Trained Model
torch.save(model.state_dict(), 'model/brain_tumor.pth')
print("🎉 Training Completed & Model Saved")
