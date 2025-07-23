# Smart Parking Space Detector

An AI-powered web application that automatically detects parking spaces in images and determines their occupancy status (free or occupied) using computer vision and deep learning.

![Demo](https://img.shields.io/badge/Demo-Live-green) ![Python](https://img.shields.io/badge/Python-3.7+-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Real-time Detection**: Automatically locates parking spaces in uploaded images
- **Occupancy Classification**: Determines if each space is free or occupied
- **Visual Results**: Color-coded bounding boxes (Green = Free, Red = Occupied)
- **Detailed Analytics**: Comprehensive statistics and confidence scores
- **Web Interface**: User-friendly drag-and-drop interface
- **High Accuracy**: 99%+ detection accuracy, 98.5%+ classification accuracy

## Demo

Upload a parking lot image and get instant results:
- ✅ **Green boxes** indicate free parking spaces
- ❌ **Red boxes** indicate occupied parking spaces
- 📊 **Statistics** show total, free, and occupied counts

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup

1. **Clone or download this repository**
```bash
git clone <repository-url>
cd smart-parking-detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model files**
   - Place your trained models in the `models/` folder:
   ```
   models/
   ├── yolo-best.pt      # YOLO detection model
   └── cnn-best.pth      # CNN classification model
   ```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
   - Navigate to: `http://127.0.0.1:10000`
   - Start uploading parking lot images!

## Usage

### Basic Usage
1. **Launch the app**: Run `python app.py`
2. **Upload image**: Drag and drop a parking lot image
3. **Analyze**: Click "Analyze Parking Spaces"
4. **View results**: See color-coded detection results

### Best Image Types
- **Parking lot overview shots** (aerial or elevated angle)
- **Multiple parking spaces visible** (5+ spaces recommended)
- **Clear parking space markings**
- **Good lighting conditions**
- **Standard image formats** (JPG, PNG)

### Avoid These Image Types
- Single car close-ups
- Street-level horizontal views
- Extremely dark or blurry images
- Images without visible parking lines

## Model Architecture

The system uses a two-stage AI pipeline:

### Stage 1: Parking Space Detection (YOLO)
- **Purpose**: Locate where parking spaces are in the image
- **Model**: YOLOv5s with custom training
- **Output**: Bounding box coordinates of parking spaces

### Stage 2: Occupancy Classification (CNN)
- **Purpose**: Determine if each space is free or occupied
- **Model**: Custom CNN with 4 convolutional layers
- **Output**: Binary classification (free/occupied) with confidence scores

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.7+

### Recommended Requirements
- **RAM**: 8GB or higher
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)
- **Internet**: For initial model downloads

## Dependencies

Core dependencies (see `requirements.txt`):
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `ultralytics` - YOLOv5 implementation
- `gradio` - Web interface framework
- `opencv-python` - Image processing
- `pillow` - Image handling
- `numpy` - Numerical computations

## File Structure

```
smart-parking-detector/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Model files directory
│   ├── yolo-best.pt     # YOLO detection model
│   └── cnn-best.pth     # CNN classification model
└── test_images/         # Sample test images (optional)
```

## Configuration

### Model Paths
Update model paths in `app.py` if needed:
```python
REQUIRED_FILES = {
    'yolo_model': './models/yolo-best.pt',
    'cnn_model': './models/cnn-best.pth'
}
```

### Server Settings
Modify server configuration:
```python
app.launch(
    share=False,           # Set to True for public URL
    server_name="127.0.0.1",  # Local access only
    server_port=10000       # Change port if needed
)
```

## Performance

### Accuracy Metrics
- **YOLO Detection**: 99%+ mAP@0.5
- **CNN Classification**: 98.56% accuracy
- **Combined System**: ~97% end-to-end accuracy

### Processing Speed
- **Typical processing time**: 2-5 seconds per image
- **Depends on**: Image size, number of parking spaces, hardware

## Troubleshooting

### Common Issues

**"No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**"YOLO loading failed"**
- Ensure model file exists in `./models/yolo-best.pt`
- Check internet connection for initial YOLOv5 download

**"CNN loading failed"**
- Verify model file exists in `./models/cnn-best.pth`
- Ensure sufficient RAM available

**Poor detection results**
- Use parking lot overview images (not close-ups)
- Ensure good lighting and clear parking lines
- Try different camera angles

### Getting Help
If you encounter issues:
1. Check that all model files are in the correct location
2. Verify all dependencies are installed correctly
3. Ensure your test images match the recommended criteria

## Technical Details

### Input Requirements
- **Image format**: JPG, PNG, or other common formats
- **Resolution**: Any size (automatically resized)
- **Perspective**: Elevated angle preferred (security camera style)

### Output Format
- **Visual**: Annotated image with colored bounding boxes
- **Statistics**: Total spaces, free count, occupied count, occupancy rate
- **Details**: Individual space classifications with confidence scores

## Applications

### Commercial Use Cases
- **Shopping centers**: Real-time parking availability displays
- **Office buildings**: Employee parking management
- **Airports**: Passenger parking guidance
- **Universities**: Campus parking optimization

### Smart City Integration
- **Traffic management**: Reduce congestion through parking guidance
- **Urban planning**: Data-driven parking infrastructure decisions
- **Environmental impact**: Reduce emissions from parking search time

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- Built using YOLOv5 by Ultralytics
- Web interface powered by Gradio
- Trained on CNRPark-EXT dataset

---

**Ready to detect parking spaces?** 🚗 Run `python app.py` and start analyzing!