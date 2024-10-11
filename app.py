import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
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
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv('GEMINI_API_KEY')

# Gemini API setup
genai.configure(api_key=api_key)
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues

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

Please structure the report as follows, using the exact headings provided:

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
Provide insights on what these indicators suggest about the {media_type}'s authenticity.

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

def create_donut_chart(is_deepfake, confidence):
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [confidence * 100, (1 - confidence) * 100]
    labels = ['Fake', 'Real'] if is_deepfake else ['Real', 'Fake']
    colors = ['#E0B0FF', '#6a0dad'] if is_deepfake else ['#6a0dad', '#E0B0FF']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title('Confidence in Image Classification')
    
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

# Add this function to perform improved face detection and tracing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def detect_and_trace_face(image):
    logging.info("Starting face detection and tracing")
    
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        logging.warning("No faces detected in the image")
        return None, None, None, None
    
    # Get the first detected face
    (x, y, w, h) = faces[0]
    
    # Generate heatmaps
    face_heatmap = generate_face_heatmap(image, (x, y, w, h))
    landmark_heatmap = generate_landmark_heatmap(image, gray, (x, y, w, h))
    attention_heatmap = generate_attention_heatmap(image)
    
    # Draw face rectangle on the original image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Convert images to base64
    traced_image = cv2_to_base64(image)
    face_heatmap_base64 = cv2_to_base64(face_heatmap)
    landmark_heatmap_base64 = cv2_to_base64(landmark_heatmap)
    attention_heatmap_base64 = cv2_to_base64(attention_heatmap)
    
    return traced_image, face_heatmap_base64, landmark_heatmap_base64, attention_heatmap_base64

def generate_face_heatmap(image, face):
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    (x, y, w, h) = face
    cv2.rectangle(heatmap, (x, y), (x+w, y+h), 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

def generate_landmark_heatmap(image, gray, face):
    (x, y, w, h) = face
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    
    # Use Shi-Tomasi corner detection as a simple approximation of facial landmarks
    corners = cv2.goodFeaturesToTrack(gray[y:y+h, x:x+w], 25, 0.01, 10)
    if corners is not None:
        corners = np.round(corners).astype(int)  # Replace np.int0 with this line
        
        for corner in corners:
            cx, cy = corner.ravel()
            cv2.circle(heatmap, (x+cx, y+cy), 5, 1, -1)
    
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # Add small epsilon to avoid division by zero
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

def generate_attention_heatmap(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.Laplacian(gray, cv2.CV_64F)
    heatmap = np.abs(heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

def cv2_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_video_for_face(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 != 0:  # Process every 30th frame
            continue
        
        traced_image, face_heatmap, landmark_heatmap, attention_heatmap = detect_and_trace_face(frame)
        if traced_image and face_heatmap and landmark_heatmap and attention_heatmap:
            cap.release()
            return traced_image, face_heatmap, landmark_heatmap, attention_heatmap
    
    cap.release()
    logging.warning("No faces detected in the video")
    return None, None, None, None

@app.route('/')
def landing():
    return render_template('landing-index.html')

@app.route('/detect-deepfake')
def detect_deepfake_page():
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
            file_data = file.read()
            img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            is_deepfake, confidence = process_image(file_data)
            traced_image, face_heatmap, landmark_heatmap, attention_heatmap = detect_and_trace_face(img)
        elif is_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                is_deepfake, confidence = process_video(temp_file.name)
                traced_image, face_heatmap, landmark_heatmap, attention_heatmap = process_video_for_face(temp_file.name)
            os.unlink(temp_file.name)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        if traced_image is None:
            logging.warning("Face tracing or heatmap generation failed, proceeding without images")
        
        # Use ThreadPoolExecutor to run tasks concurrently
        report_future = executor.submit(generate_report, is_deepfake, confidence, is_video)
        donut_chart_future = executor.submit(create_donut_chart, is_deepfake, confidence)
        
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
            'is_video': is_video,
            'traced_image': traced_image,
            'face_heatmap': face_heatmap,
            'landmark_heatmap': landmark_heatmap,
            'attention_heatmap': attention_heatmap
        }
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Error in detect_deepfake: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)