import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def train_model(model, train_loader, criterion, optimizer, epochs=10, model_save_path='model/brain_tumor.pth'):
    """
    Train the brain tumor detection model.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for training
        epochs: Number of training epochs
        model_save_path: Path to save the trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()  # Set model to training mode
    
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
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"🎉 Training Completed & Model Saved at {model_save_path}")

# If running as standalone script
if __name__ == '__main__':
    from torchvision import datasets, transforms
    try:
        from src.model import BrainTumorCNN
    except ImportError:
        # Fallback for different import paths
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.model import BrainTumorCNN
    
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
    dataset = datasets.ImageFolder(root='../dataset/', transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 🧠 Model Init
    model = BrainTumorCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=epochs)
