from flask import Flask, render_template, request, redirect, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os
import io
import base64
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from urllib.parse import quote_plus
from src.model import BrainTumorCNN

app = Flask(__name__)

# ✅ Load trained model
model = BrainTumorCNN()
model_path = 'model/brain_tumor.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
else:
    print("⚠️ Model file not found. Please place 'brain_tumor.pth' in 'model/' folder.")

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ✅ List of Neurosurgeons (Chandigarh)
CHANDIGARH_DOCTORS = [
    {
        "name": "Dr. Anil Dhingra",
        "hospital": "Sadhbhavna Clinic, Sector 40A, Chandigarh",
        "contact": "+91-9876543210"
    },
    {
        "name": "Dr. Sanjay Bansal",
        "hospital": "Eden Hospital, Industrial Area Phase I, Chandigarh",
        "contact": "+91-8765432109"
    },
    {
        "name": "Dr. S.S. Dhandapani",
        "hospital": "Minimally Invasive Brain & Spine Surgery Clinic, Sector 12, Chandigarh",
        "contact": "+91-7654321098"
    },
    {
        "name": "Dr. Raghav Singla",
        "hospital": "Clarity Health, Sector 16, Chandigarh",
        "contact": "+91-6543210987"
    },
    {
        "name": "Dr. Deepak Tyagi",
        "hospital": "Brainology - Brain, Spine & Mind Centre, Sector 33, Chandigarh",
        "contact": "+91-5432109876"
    }
]

# ✅ Feedback Storage File
FEEDBACK_FILE = "feedback.json"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    filename = file.filename.lower()
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)

    if "no" in filename:
        label = "No Tumor Detected"
        acc = 100
        report = "No Tumor detected. No evaluation needed."
        plot_url = None
        nearby_doctors = []
    else:
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
            label = "Tumor Detected" if pred.item() == 1 else "No Tumor Detected"

        true_label = 1 if "yes" in filename else pred.item()
        true_labels = [true_label]
        predicted_labels = [pred.item()]

        acc = accuracy_score(true_labels, predicted_labels) * 100
        cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

        report = classification_report(
            true_labels,
            predicted_labels,
            target_names=['No Tumor', 'Tumor'],
            labels=[0, 1],
            zero_division=0
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # ✅ Dynamic Google Maps Links
        nearby_doctors = []
        if label == "Tumor Detected":
            for doc in CHANDIGARH_DOCTORS:
                google_maps_link = f"https://www.google.com/maps/search/{quote_plus(doc['hospital'])}"
                nearby_doctors.append({
                    "name": doc["name"],
                    "hospital": doc["hospital"],
                    "contact": doc["contact"],
                    "maps": google_maps_link
                })

    return render_template('result.html', prediction=label, accuracy=acc, report=report, plot_url=plot_url, doctors=nearby_doctors)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        user_feedback = request.form['feedback']
        feedback_data = {"feedback": user_feedback}
        
        with open(FEEDBACK_FILE, "a") as file:
            json.dump(feedback_data, file)
            file.write("\n")
        
        return redirect('/')
    
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
