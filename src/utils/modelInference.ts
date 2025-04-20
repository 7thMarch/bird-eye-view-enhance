
import * as ort from 'onnxruntime-web';
import { BirdNestDetection, PreprocessedImageData } from '@/types';

// Configure ONNX Runtime web
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

// Class names for the model
const classNames = ['birdnest'];

/**
 * Initialize the ONNX model
 */
export const initModel = async (modelUrl: string): Promise<ort.InferenceSession> => {
  try {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    return session;
  } catch (error) {
    console.error('Failed to initialize the model:', error);
    throw error;
  }
};

/**
 * Run inference with the YOLOv8 model
 */
export const runInference = async (
  session: ort.InferenceSession,
  imageData: PreprocessedImageData,
  confidenceThreshold: number = 0.25,
  iouThreshold: number = 0.45
): Promise<BirdNestDetection[]> => {
  try {
    // Prepare input tensor
    const inputTensor = new ort.Tensor(
      'float32',
      imageData.tensor,
      [1, 3, imageData.height, imageData.width] // NCHW format
    );

    // Run inference
    const feeds = { images: inputTensor };
    const results = await session.run(feeds);
    
    // Process results
    // YOLOv8 typically outputs with shape [1, num_detections, 5+num_classes]
    // Where each detection is [x, y, width, height, confidence, class_scores...]
    const output = results.output;
    
    if (!output) {
      throw new Error('Model output is undefined');
    }
    
    return processYoloOutput(
      output,
      imageData.originalWidth,
      imageData.originalHeight,
      imageData.width,
      imageData.height,
      confidenceThreshold
    );
  } catch (error) {
    console.error('Inference failed:', error);
    throw error;
  }
};

/**
 * Process the raw YOLOv8 output into usable detections
 */
function processYoloOutput(
  outputTensor: ort.Tensor,
  originalWidth: number,
  originalHeight: number,
  modelWidth: number,
  modelHeight: number,
  confidenceThreshold: number
): BirdNestDetection[] {
  const data = outputTensor.data as Float32Array;
  const dimensions = outputTensor.dims;
  
  // YOLOv8 output shape is typically [1, num_detections, 5+num_classes]
  const numDetections = dimensions[1];
  const detectionSize = dimensions[2];
  
  const detections: BirdNestDetection[] = [];
  
  // Scale factors to map back to original image
  const xScale = originalWidth / modelWidth;
  const yScale = originalHeight / modelHeight;
  
  for (let i = 0; i < numDetections; i++) {
    const baseIndex = i * detectionSize;
    
    // YOLOv8 outputs center coordinates, width and height, and confidence + class scores
    const x = data[baseIndex];
    const y = data[baseIndex + 1];
    const width = data[baseIndex + 2];
    const height = data[baseIndex + 3];
    const confidence = data[baseIndex + 4];
    
    // Only consider detections with confidence above threshold
    if (confidence < confidenceThreshold) continue;
    
    // Find the class with highest probability
    let maxClassProb = 0;
    let classId = 0;
    
    for (let c = 0; c < classNames.length; c++) {
      const classProb = data[baseIndex + 5 + c];
      if (classProb > maxClassProb) {
        maxClassProb = classProb;
        classId = c;
      }
    }
    
    // Calculate final score
    const score = confidence * maxClassProb;
    
    // Skip low-confidence detections
    if (score < confidenceThreshold) continue;
    
    // Convert from center coordinates to top-left coordinates and scale to original image
    const scaledX = (x - width / 2) * xScale;
    const scaledY = (y - height / 2) * yScale;
    const scaledWidth = width * xScale;
    const scaledHeight = height * yScale;
    
    detections.push({
      bbox: [scaledX, scaledY, scaledWidth, scaledHeight],
      score,
      class_id: classId,
      class_name: classNames[classId]
    });
  }
  
  return detections;
}
