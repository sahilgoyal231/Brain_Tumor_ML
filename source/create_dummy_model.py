# 📁 create_dummy_model.py

import torch
import torch.nn as nn

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 🔄 Adaptive Pooling for consistent size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed to 4x4

        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # (64 * 4 * 4 = 1024)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)  # 🔄 Adaptive Pooling
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Initialize dummy model
model = BrainTumorCNN()

# 💾 Save the dummy model as brain_tumor.pth
torch.save(model.state_dict(), '../model/brain_tumor.pth')

print("🎉 Dummy BrainTumorCNN saved at model/brain_tumor.pth")
