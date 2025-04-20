
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
    // Get the first output from the model
    const outputKey = Object.keys(results)[0];
    const output = results[outputKey];
    
    if (!output) {
      throw new Error('Model output is undefined');
    }
    
    console.log('Model output shape:', output.dims);
    console.log('Model output type:', output.type);
    
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
  
  console.log('Processing output with dimensions:', dimensions);
  
  // YOLOv8 output shape could be different depending on export parameters
  // Common formats are [1, num_detections, 5+num_classes] or [1, 84, num_boxes]
  let detections: BirdNestDetection[] = [];
  
  // Scale factors to map back to original image
  const xScale = originalWidth / modelWidth;
  const yScale = originalHeight / modelHeight;
  
  // Handle YOLOv8 default output format (3 possible formats)
  
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
  // Format 2: [1, 84, anchor_grid_size, anchor_grid_size]
  // YOLOv8 new detection format
  else if (dimensions.length === 4) {
    console.log('Processing YOLOv8 grid format output');
    
    const numClasses = classNames.length;
    const channels = dimensions[1];
    const gridHeight = dimensions[2];
    const gridWidth = dimensions[3];
    
    // For YOLOv8, typically the channels would be 4 (box) + 1 (confidence) + numClasses
    const boxOffset = 0;  // x, y, w, h are the first 4 channels
    const confidenceOffset = 4;  // confidence is the 5th channel
    const classOffset = 5;  // classes start from the 6th channel
    
    // Process each grid cell
    for (let cy = 0; cy < gridHeight; cy++) {
      for (let cx = 0; cx < gridWidth; cx++) {
        // Get confidence score
        const confidenceIndex = getIndexFrom4D(dimensions, 0, confidenceOffset, cy, cx);
        const confidence = data[confidenceIndex];
        
        if (confidence < confidenceThreshold) continue;
        
        // Find the class with highest probability
        let maxClassProb = 0;
        let classId = 0;
        
        for (let c = 0; c < numClasses; c++) {
          const classIndex = getIndexFrom4D(dimensions, 0, classOffset + c, cy, cx);
          const classProb = data[classIndex];
          
          if (classProb > maxClassProb) {
            maxClassProb = classProb;
            classId = c;
          }
        }
        
        // Calculate final score
        const score = confidence * maxClassProb;
        
        // Skip low-confidence detections
        if (score < confidenceThreshold) continue;
        
        // Get bounding box coordinates
        const xIndex = getIndexFrom4D(dimensions, 0, boxOffset, cy, cx);
        const yIndex = getIndexFrom4D(dimensions, 0, boxOffset + 1, cy, cx);
        const wIndex = getIndexFrom4D(dimensions, 0, boxOffset + 2, cy, cx);
        const hIndex = getIndexFrom4D(dimensions, 0, boxOffset + 3, cy, cx);
        
        const x = data[xIndex];
        const y = data[yIndex];
        const width = data[wIndex];
        const height = data[hIndex];
        
        // Convert from grid to image coordinates and scale to original image
        const gridCellSize = 1.0 / Math.max(gridWidth, gridHeight);
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
  // Format 3: Direct output [1, 84, num_boxes]
  else if (dimensions.length === 3 && dimensions[1] === 84) {
    console.log('Processing direct YOLOv8 output format');
    
    const numBoxes = dimensions[2];
    
    for (let i = 0; i < numBoxes; i++) {
      const x = data[0 * numBoxes + i];
      const y = data[1 * numBoxes + i];
      const width = data[2 * numBoxes + i];
      const height = data[3 * numBoxes + i];
      const confidence = data[4 * numBoxes + i];
      
      if (confidence < confidenceThreshold) continue;
      
      // Assume it's the only class we care about
      const score = confidence;
      
      // Convert from normalized coordinates to original image coordinates
      const scaledX = (x - width / 2) * xScale;
      const scaledY = (y - height / 2) * yScale;
      const scaledWidth = width * xScale;
      const scaledHeight = height * yScale;
      
      detections.push({
        bbox: [scaledX, scaledY, scaledWidth, scaledHeight],
        score,
        class_id: 0,
        class_name: classNames[0]
      });
    }
  }
  // Last resort - try to handle YOLOv8 output dynamically
  else {
    console.log('Trying to process unknown YOLOv8 output format');
    
    try {
      const flatData = Array.from(data);
      let batchDetections = [];
      
      // Try to find detection patterns in the flat array
      for (let i = 0; i < flatData.length; i += 6) {  // Assuming [x,y,w,h,conf,class_id]
        if (i + 5 < flatData.length) {
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
            
            batchDetections.push({
              bbox: [scaledX, scaledY, scaledWidth, scaledHeight],
              score: confidence,
              class_id: classId,
              class_name: classNames[Math.min(classId, classNames.length - 1)]
            });
          }
        }
      }
      
      if (batchDetections.length > 0) {
        detections = batchDetections;
      }
    } catch (err) {
      console.error('Failed to process unknown output format:', err);
    }
  }
  
  console.log('Processed detections:', detections);
  return detections;
}

// Helper function to get an index in a 4D tensor
function getIndexFrom4D(
  dimensions: readonly number[],
  n: number,
  c: number,
  h: number,
  w: number
): number {
  return n * dimensions[1] * dimensions[2] * dimensions[3] +
         c * dimensions[2] * dimensions[3] +
         h * dimensions[3] +
         w;
}
