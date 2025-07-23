import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image as PILImage
import gradio as gr
from pathlib import Path

print("SMART PARKING DEMO APP")
print("="*40)

# Required files
REQUIRED_FILES = {
    'yolo_model': './models/yolo-best.pt',
    'cnn_model': './models/cnn-best.pth'
}

print("Checking required files...")
for name, path in REQUIRED_FILES.items():
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {path}")

missing_files = [name for name, path in REQUIRED_FILES.items() if not Path(path).exists()]

if missing_files:
    print(f"❌ Missing files: {missing_files}")
    exit()
else:
    print("✅ All required files found!")

# CNN Architecture
class ParkingSpotCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ParkingSpotCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Load models
print("Loading models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load YOLO
print("Loading YOLO model...")
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                               path=REQUIRED_FILES['yolo_model'], 
                               force_reload=True, trust_repo=True)
    yolo_model.conf = 0.25
    yolo_model.iou = 0.45
    print("✓ YOLO model loaded")
except Exception as e:
    print(f"✗ YOLO loading failed: {e}")
    yolo_model = None

# Load CNN
print("Loading CNN model...")
try:
    cnn_model = ParkingSpotCNN(num_classes=2).to(device)
    cnn_model.load_state_dict(torch.load(REQUIRED_FILES['cnn_model'], map_location=device))
    cnn_model.eval()
    print("✓ CNN model loaded")
except Exception as e:
    print(f"✗ CNN loading failed: {e}")
    cnn_model = None

# CNN preprocessing
cnn_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Smart Parking Detector
class SmartParkingDetector:
    def __init__(self, yolo_model, cnn_model, cnn_transform, device):
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.cnn_transform = cnn_transform
        self.device = device
        self.class_names = ['free', 'occupied']
        
    def detect_parking_spaces(self, image):
        if self.yolo_model is None:
            return []
            
        results = self.yolo_model(image)
        detections = []
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class': int(cls)
            })
        
        return detections
    
    def classify_occupancy(self, image, bbox):
        if self.cnn_model is None:
            return 'unknown', 0.0
            
        x1, y1, x2, y2 = bbox
        patch = image[y1:y2, x1:x2]
        
        if patch.size == 0:
            return 'unknown', 0.0
        
        patch_pil = PILImage.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        patch_tensor = self.cnn_transform(patch_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.cnn_model(patch_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        class_name = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        return class_name, confidence_score
    
    def process_image(self, image):
        if isinstance(image, PILImage.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, str):
            image = cv2.imread(image)
            
        if image is None:
            return None, None
            
        detections = self.detect_parking_spaces(image)
        
        results = []
        for detection in detections:
            occupancy, confidence = self.classify_occupancy(image, detection['bbox'])
            
            result = {
                'bbox': detection['bbox'],
                'yolo_confidence': detection['confidence'],
                'occupancy': occupancy,
                'occupancy_confidence': confidence,
                'combined_confidence': detection['confidence'] * confidence
            }
            results.append(result)
        
        return results, image
    
    def visualize_results(self, image, results):
        vis_image = image.copy()
        
        free_count = 0
        occupied_count = 0
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            occupancy = result['occupancy']
            
            if occupancy == 'free':
                color = (0, 255, 0)
                free_count += 1
            elif occupancy == 'occupied':
                color = (0, 0, 255)
                occupied_count += 1
            else:
                color = (0, 255, 255)
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
            
            label = f"{occupancy.upper()}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        total_spots = len(results)
        summary = f"TOTAL: {total_spots} | FREE: {free_count} | OCCUPIED: {occupied_count}"
        
        cv2.putText(vis_image, summary, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        cv2.putText(vis_image, summary, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return vis_image, {
            'total': total_spots,
            'free': free_count,
            'occupied': occupied_count,
            'occupancy_rate': (occupied_count / total_spots * 100) if total_spots > 0 else 0
        }

# Initialize detector
if yolo_model is not None and cnn_model is not None:
    detector = SmartParkingDetector(yolo_model, cnn_model, cnn_transform, device)
    print("✅ Smart Parking Detector initialized!")
else:
    print("❌ Cannot initialize detector - models not loaded properly")
    detector = None

# Gradio processing function
def process_uploaded_image(image):
    if detector is None:
        return None, "❌ Models not loaded properly"
    
    try:
        results, _ = detector.process_image(image)
        
        if results is None:
            return None, "❌ Failed to process image"
        
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        vis_image, stats = detector.visualize_results(image_cv, results)
        
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        vis_image_pil = PILImage.fromarray(vis_image_rgb)
        
        report = f"""SMART PARKING ANALYSIS

SUMMARY:
• Total Spaces: {stats['total']}
• Free Spaces: {stats['free']}
• Occupied Spaces: {stats['occupied']}
• Occupancy Rate: {stats['occupancy_rate']:.1f}%

DETAILS:
"""
        
        for i, result in enumerate(results[:10]):
            occupancy = result['occupancy']
            yolo_conf = result['yolo_confidence']
            occ_conf = result['occupancy_confidence']
            
            status = "FREE" if occupancy == 'free' else "OCCUPIED"
            report += f"• Space {i+1}: {status} (YOLO: {yolo_conf:.2f}, CNN: {occ_conf:.2f})\n"
        
        if len(results) > 10:
            report += f"... and {len(results) - 10} more spaces\n"
        
        return vis_image_pil, report
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Create Gradio app
def create_app():
    with gr.Blocks(title="Smart Parking Detector") as app:
        
        gr.HTML("""
        <div style="text-align: center; background: linear-gradient(90deg, #4CAF50, #2196F3); 
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1>Smart Parking Space Detector</h1>
            <p>Upload a parking lot image to detect and analyze parking space occupancy</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Image")
                input_image = gr.Image(type="pil", label="Parking Lot Image")
                process_btn = gr.Button("Analyze Parking Spaces", variant="primary")
                
            with gr.Column():
                gr.Markdown("### Results")
                output_image = gr.Image(label="Detected Parking Spaces")
                output_report = gr.Textbox(label="Analysis Report", lines=15)
        
        gr.Markdown("""
        ### Instructions:
        1. Upload a parking lot image
        2. Click "Analyze Parking Spaces"  
        3. View results: Green = Free, Red = Occupied
        """)
        
        process_btn.click(
            fn=process_uploaded_image,
            inputs=input_image,
            outputs=[output_image, output_report]
        )
    
    return app

# Launch app
if detector is not None:
    print("🚀 Launching web application...")
    app = create_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=10000)
else:
    print("❌ Cannot launch app - detector not initialized")

print("✅ DEMO APP COMPLETE!")