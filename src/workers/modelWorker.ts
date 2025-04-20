
import * as ort from 'onnxruntime-web';
import { PreprocessedImageData, BirdNestDetection } from '../types';

// This file will be used as a web worker to process images in a separate thread

// Configure ONNX Runtime web
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

// Class names for the model
const classNames = ['birdnest'];

// Define worker message types
type WorkerMessageIn = {
  type: 'LOAD_MODEL' | 'RUN_INFERENCE';
  modelUrl?: string;
  imageData?: PreprocessedImageData;
  confidenceThreshold?: number;
  iouThreshold?: number;
};

type WorkerMessageOut = {
  type: 'MODEL_LOADED' | 'INFERENCE_RESULT' | 'ERROR';
  detections?: BirdNestDetection[];
  error?: string;
};

let session: ort.InferenceSession | null = null;

// Handle incoming messages from the main thread
self.onmessage = async (event: MessageEvent<WorkerMessageIn>) => {
  const { type } = event.data;
  
  try {
    switch (type) {
      case 'LOAD_MODEL':
        if (!event.data.modelUrl) {
          throw new Error('Model URL is required');
        }
        
        // Initialize the model
        session = await ort.InferenceSession.create(event.data.modelUrl, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        
        // Notify that the model is loaded
        const modelLoadedMessage: WorkerMessageOut = { type: 'MODEL_LOADED' };
        self.postMessage(modelLoadedMessage);
        break;
        
      case 'RUN_INFERENCE':
        if (!session) {
          throw new Error('Model is not loaded');
        }
        
        if (!event.data.imageData) {
          throw new Error('Image data is required');
        }
        
        const imageData = event.data.imageData;
        const confidenceThreshold = event.data.confidenceThreshold || 0.25;
        const iouThreshold = event.data.iouThreshold || 0.45;
        
        // Run inference
        const detections = await runInference(
          session,
          imageData,
          confidenceThreshold,
          iouThreshold
        );
        
        // Send back the results
        const inferenceResultMessage: WorkerMessageOut = {
          type: 'INFERENCE_RESULT',
          detections
        };
        self.postMessage(inferenceResultMessage);
        break;
        
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    // Handle any errors
    const errorMessage: WorkerMessageOut = {
      type: 'ERROR',
      error: error instanceof Error ? error.message : String(error)
    };
    self.postMessage(errorMessage);
  }
};

/**
 * Run inference with the YOLOv8 model
 */
async function runInference(
  session: ort.InferenceSession,
  imageData: PreprocessedImageData,
  confidenceThreshold: number,
  iouThreshold: number
): Promise<BirdNestDetection[]> {
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
}

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
