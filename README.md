
# Bird Nest Detector Web App

A browser-based application for detecting bird nests in images using YOLOv8, running entirely in the browser with ONNX Runtime Web.

## Features

- **Browser-Based Detection**: Run a YOLOv8 model directly in your browser
- **No Server Required**: All processing happens locally on your device
- **Privacy First**: Your images never leave your device
- **WebGL/WebGPU Acceleration**: Uses hardware acceleration when available
- **Easy to Use**: Simple drag-and-drop interface for image uploads

## Technologies Used

- **React**: For building the UI
- **TypeScript**: For type safety
- **Tailwind CSS**: For styling
- **ONNX Runtime Web**: For running the ML model in the browser
- **Web Workers**: For non-blocking inference

## Setup and Run

1. Clone this repository:
```bash
git clone <repository-url>
cd bird-nest-detector
```

2. Install dependencies:
```bash
npm install
```

3. Add your model:
   - Place your YOLOv8 ONNX model in the `public/models/` directory with the name `birdnest.onnx`.
   - For sample images, place them in the `public/images/` directory.

4. Start the development server:
```bash
npm run dev
```

5. Open your browser and navigate to `http://localhost:8080`

## Building for Production

To create a production build:

```bash
npm run build
```

The build files will be in the `dist/` directory.

## How It Works

1. **Model Loading**: The application loads a YOLOv8 model converted to ONNX format.
2. **Image Processing**: Uploaded images are preprocessed to match the model's input requirements.
3. **Inference**: The model runs in a Web Worker to prevent UI blocking.
4. **Visualization**: Detected bird nests are displayed as bounding boxes on the image.

## Model Conversion

To convert your YOLOv8 model to ONNX format:

```python
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('birdnest.pt')

# Export to ONNX format
model.export(format='onnx', dynamic=True, simplify=True)
```

## Performance Considerations

- **WebGPU**: For best performance, use Chrome with WebGPU enabled (`chrome://flags/#enable-unsafe-webgpu`).
- **Model Size**: Smaller models will load faster and perform better on less powerful devices.
- **Image Size**: Large images will be resized to fit the model's input dimensions.

## Credits

- Inspired by [web-realesrgan](https://github.com/xororz/web-realesrgan) project
- Uses [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) for model inference
