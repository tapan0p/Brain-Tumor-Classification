from flask import Flask, render_template, request, jsonify
from model import BrainTumorClassifier, inference_pipeline
import torch
import os

app = Flask(__name__)

model_path = os.path.join('..', 'Model', 'brain_tumor_classifier.pth')
model = BrainTumorClassifier(num_classes=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path,map_location=device))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    prediction, confidence = inference_pipeline(image, model)
    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
    })

if __name__ == '__main__':
    app.run(debug=True)
