
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
    console.error('Worker error:', error);
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
  console.log('Model inputs:', session.inputNames);
  console.log('Model outputs:', session.outputNames);
  
  // Try different input names if 'images' doesn't work
  const inputName = session.inputNames[0];
  const feedsWithCorrectName = { [inputName]: inputTensor };
  
  const results = await session.run(feedsWithCorrectName);
  
  // Get the first output from the model
  const outputName = session.outputNames[0];
  const output = results[outputName];
  
  console.log('Output tensor:', output);
  console.log('Output shape:', output?.dims);
  
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
  
  console.log('Processing output with dimensions:', dimensions);
  
  // YOLOv8 output shape could vary based on export options
  let detections: BirdNestDetection[] = [];
  
  // Scale factors to map back to original image
  const xScale = originalWidth / modelWidth;
  const yScale = originalHeight / modelHeight;
  
  // Format 1: [1, num_detections, 5+num_classes] (standard format)
  if (dimensions.length === 3 && dimensions[2] > 5) {
    const numDetections = dimensions[1];
    const detectionSize = dimensions[2];
    
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
  }
  // Last resort - try to handle YOLOv8 output based on common patterns
  else {
    console.log('Trying to process alternative YOLOv8 output format');
    
    try {
      // YOLOv8 can output in various formats based on export parameters
      // This is a simplified approach to handle common formats
      
      // If we're dealing with a flat array of detections [x1,y1,w1,h1,conf1,cls1,x2,y2,...]
      if (dimensions.length === 1 || (dimensions.length === 2 && dimensions[0] === 1)) {
        const flatData = Array.from(data);
        const boxSize = 6; // [x,y,w,h,conf,cls]
        
        for (let i = 0; i < flatData.length; i += boxSize) {
          if (i + boxSize <= flatData.length) {
            const confidence = flatData[i + 4];
            
            if (confidence >= confidenceThreshold) {
              const x = flatData[i];
              const y = flatData[i + 1];
              const width = flatData[i + 2];
              const height = flatData[i + 3];
              const classId = Math.round(flatData[i + 5]);
              
              // Convert to original image coordinates
              const scaledX = (x - width / 2) * xScale;
              const scaledY = (y - height / 2) * yScale;
              const scaledWidth = width * xScale;
              const scaledHeight = height * yScale;
              
              detections.push({
                bbox: [scaledX, scaledY, scaledWidth, scaledHeight],
                score: confidence,
                class_id: classId,
                class_name: classNames[Math.min(classId, classNames.length - 1)]
              });
            }
          }
        }
      }
      // If we're dealing with a batch of detections [batch_size, num_detections, xywh+conf+classes]
      else if (dimensions.length === 3) {
        const batchSize = dimensions[0];
        const numDetections = dimensions[1];
        const elementsPerDetection = dimensions[2];
        
        for (let b = 0; b < batchSize; b++) {
          for (let i = 0; i < numDetections; i++) {
            const baseIdx = b * numDetections * elementsPerDetection + i * elementsPerDetection;
            
            // YOLOv8 standard output format has confidence at index 4
            const confidence = data[baseIdx + 4];
            
            if (confidence >= confidenceThreshold) {
              const x = data[baseIdx];
              const y = data[baseIdx + 1];
              const width = data[baseIdx + 2];
              const height = data[baseIdx + 3];
              
              // Find max class probability
              let maxClassProb = 0;
              let classId = 0;
              for (let c = 0; c < classNames.length; c++) {
                const classIdx = baseIdx + 5 + c;
                if (classIdx < baseIdx + elementsPerDetection) {
                  const classProb = data[classIdx];
                  if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classId = c;
                  }
                }
              }
              
              // Calculate final score
              const score = confidence * (maxClassProb > 0 ? maxClassProb : 1);
              
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
          }
        }
      }
    } catch (err) {
      console.error('Failed to process alternative output format:', err);
    }
  }
  
  console.log('Processed detections:', detections);
  return detections;
}
