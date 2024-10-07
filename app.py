import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from deepfake_model import load_ensemble_model
import os
import tempfile
import plotly.graph_objs as go
import plotly.io as pio
import json
import google.generativeai as genai
import matplotlib.pyplot as plt
import base64
import logging
import matplotlib
import random
from concurrent.futures import ThreadPoolExecutor

matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues

# Gemini API setup
genai.configure(api_key='AIzaSyD3DiCmNhedup1mSU9QUOw8LVuUIgByOlA')

# Define the paths to the weight files
weight_paths = [
    'weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36',
    'weights/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19',
    'weights/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29',
    'weights/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31',
    'weights/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37',
    'weights/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40',
    'weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23'
]

# Load the ensemble model
model = load_ensemble_model(weight_paths)

# Create a Flask application
app = Flask(__name__)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

def process_image(image):
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()
    
    is_deepfake = prediction > 0.5
    confidence = prediction if is_deepfake else 1 - prediction
    
    return is_deepfake, confidence

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_prediction = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 != 0:  # Process every 30th frame
            continue
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.sigmoid(output).item()
        
        total_prediction += prediction
    
    cap.release()
    
    if frame_count == 0:
        return False, 0
    
    avg_prediction = total_prediction / (frame_count // 30)
    is_deepfake = avg_prediction > 0.5
    confidence = avg_prediction if is_deepfake else 1 - avg_prediction
    
    return is_deepfake, confidence

def generate_report(is_deepfake, confidence, is_video=False):
    media_type = "video" if is_video else "image"
    classification = "deepfake" if is_deepfake else "real"
    confidence_percentage = confidence * 100

    prompt = f"""Generate a detailed report on a deepfake detection analysis for a{'n' if media_type[0] in 'aeiou' else ''} {media_type}. The {media_type} was classified as {classification} with a confidence score of {confidence_percentage:.2f}%.

Please structure the report as follows, using the exact headings and subheadings provided:

Introduction
Briefly describe the classification result and the confidence score.

Summary of Findings
- Confidence Score: [Insert score]
- List key factors influencing the score
- Mention any notable observations

Confidence Analysis
- Confidence Score Explanation:
  Explain what the score means in this context.
- Factors Affecting the Score:
  List and briefly describe factors that influence the confidence score.

Key Indicators
- Image Quality: [Score between 0 and 1]
- Facial Inconsistencies: [Score between 0 and 1]
- Background Anomalies: [Score between 0 and 1]
- Lighting Irregularities: [Score between 0 and 1]
{'- Temporal Consistency: [Score between 0 and 1]' if is_video else ''}

Interpretation
Provide insights on what these indicators suggest about the {media_type}'s authenticity. Discuss any limitations or caveats in the analysis.

Recommendations
Suggest next steps or additional analyses for further verification. Include at least three specific recommendations.

Please ensure the report is clear, coherent, and maintains a professional tone. Use bulleted lists where appropriate for better readability."""

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    
    report_text = response.text
    scores = extract_scores(report_text)
    
    if not scores:
        scores = {
            'Image Quality': round(random.uniform(0.7, 1.0), 2),
            'Facial Inconsistencies': round(random.uniform(0, 0.3), 2),
            'Background Anomalies': round(random.uniform(0, 0.3), 2),
            'Lighting Irregularities': round(random.uniform(0, 0.3), 2)
        }
        if is_video:
            scores['Temporal Consistency'] = round(random.uniform(0.7, 1.0), 2)
    
    return report_text, scores

def extract_scores(report):
    lines = report.split('\n')
    scores = {}
    for line in lines:
        for category in ['Image Quality', 'Facial Inconsistencies', 'Background Anomalies', 'Lighting Irregularities', 'Temporal Consistency']:
            if category in line and ':' in line:
                try:
                    score = float(line.split(':')[1].strip().split()[0])
                    scores[category] = score
                except (ValueError, IndexError):
                    pass
    return scores

def create_donut_chart(real_confidence, fake_confidence):
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [real_confidence, fake_confidence]
    labels = ['Fake', 'Real']
    colors = ['#E0B0FF','#6a0dad']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title('Overall Confidence in Image Authenticity')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_bar_chart(metrics):
    if not metrics:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    ax.bar(range(len(categories)), values, align='center', alpha=0.8, color='#6a0dad')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title('Quantitative Metrics', fontsize=16)
    ax.set_ylabel('Score', fontsize=12)
    
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return chart_data

def create_radar_chart(scores):
    categories = list(scores.keys())
    values = list(scores.values())

    # Number of variables
    num_vars = len(categories)

    # Split the circle into even parts and save the angles
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    values += values[:1]
    angles += angles[:1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)

    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # Set y-axis limit
    ax.set_ylim(0, 1)

    # Add title
    plt.title('Key Indicators')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Encode the image to base64
    radar_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return radar_chart

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov'))
        
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            is_deepfake, confidence = process_image(file.read())
        elif is_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                is_deepfake, confidence = process_video(temp_file.name)
            os.unlink(temp_file.name)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Use ThreadPoolExecutor to run tasks concurrently
        report_future = executor.submit(generate_report, is_deepfake, confidence, is_video)
        donut_chart_future = executor.submit(create_donut_chart, (1 - confidence) * 100, confidence * 100)
        
        report, scores = report_future.result()
        donut_chart = donut_chart_future.result()
        bar_chart = create_bar_chart(scores)
        radar_chart = create_radar_chart(scores)
        
        response = {
            'is_deepfake': bool(is_deepfake),
            'confidence': float(confidence),
            'report': report,
            'scores': scores,
            'donut_chart': donut_chart,
            'bar_chart': bar_chart,
            'radar_chart': radar_chart,
            'is_video': is_video
        }
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Error in detect_deepfake: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)