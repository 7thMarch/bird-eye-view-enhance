
# Model Directory

Place your ONNX model file here with the name `birdnest.onnx`.

## Instructions

1. For YOLOv8, you can convert your PyTorch model (.pt) to ONNX format using:

```python
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('birdnest.pt')

# Export to ONNX format
model.export(format='onnx', dynamic=True, simplify=True)
```

2. Copy the resulting `birdnest.onnx` file to this directory.

## Model Information

The ONNX model should be a YOLOv8 model trained to detect bird nests. The expected input is an RGB image with dimensions `[1, 3, 640, 640]` and pixel values normalized to [0,1].
