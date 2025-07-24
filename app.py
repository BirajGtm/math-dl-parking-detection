import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image as PILImage
import gradio as gr
from pathlib import Path
import os

REQUIRED_FILES = {
    'yolo_model': './models/yolo-best.pt',
    'cnn_model': './models/cnn-best.pth'
}

for name, path in REQUIRED_FILES.items():
    if not Path(path).exists():
        print(f"Missing: {path}")
        exit()

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                           path=REQUIRED_FILES['yolo_model'], 
                           force_reload=True, trust_repo=True)
yolo_model.conf = 0.25
yolo_model.iou = 0.45

cnn_model = ParkingSpotCNN(num_classes=2).to(device)
cnn_model.load_state_dict(torch.load(REQUIRED_FILES['cnn_model'], map_location=device))
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SmartParkingDetector:
    def __init__(self, yolo_model, cnn_model, cnn_transform, device):
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.cnn_transform = cnn_transform
        self.device = device
        self.class_names = ['free', 'occupied']
        
    def detect_parking_spaces(self, image):
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
                'occupancy_confidence': confidence
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
            cv2.putText(vis_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        total_spots = len(results)
        summary = f"TOTAL: {total_spots} | FREE: {free_count} | OCCUPIED: {occupied_count}"
        
        cv2.putText(vis_image, summary, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return vis_image, {
            'total': total_spots,
            'free': free_count,
            'occupied': occupied_count,
            'occupancy_rate': (occupied_count / total_spots * 100) if total_spots > 0 else 0
        }

detector = SmartParkingDetector(yolo_model, cnn_model, cnn_transform, device)

def process_uploaded_image(image):
    try:
        results, _ = detector.process_image(image)
        
        if results is None:
            return None, "Failed to process image"
        
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        vis_image, stats = detector.visualize_results(image_cv, results)
        
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        vis_image_pil = PILImage.fromarray(vis_image_rgb)
        
        report = f"""PARKING ANALYSIS

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
            report += f"• Space {i+1}: {status} ({yolo_conf:.2f}, {occ_conf:.2f})\n"
        
        if len(results) > 10:
            report += f"... and {len(results) - 10} more spaces\n"
        
        return vis_image_pil, report
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_app():
    with gr.Blocks(title="Smart Parking Detector") as app:
        
        gr.HTML("""
        <div style="text-align: center; background: linear-gradient(90deg, #4CAF50, #2196F3); 
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1>Smart Parking Detector</h1>
            <p>Upload parking lot image for analysis</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Parking Lot Image")
                process_btn = gr.Button("Analyze", variant="primary")
                
            with gr.Column():
                output_image = gr.Image(label="Results")
                output_report = gr.Textbox(label="Report", lines=15)
        
        process_btn.click(
            fn=process_uploaded_image,
            inputs=input_image,
            outputs=[output_image, output_report]
        )
    
    return app

app = create_app()

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),
    share=False,
    quiet=True
)