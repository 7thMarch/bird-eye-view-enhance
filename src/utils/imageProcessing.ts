
import { PreprocessedImageData } from "@/types";

/**
 * Pre-processes an image for YOLOv8 model input
 * @param imageSrc The image source URL
 * @param modelWidth The input width expected by the model (default: 640)
 * @param modelHeight The input height expected by the model (default: 640)
 */
export const preprocessImage = async (
  imageSrc: string,
  modelWidth: number = 640,
  modelHeight: number = 640
): Promise<PreprocessedImageData> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Create a canvas to resize the image
      const canvas = document.createElement("canvas");
      canvas.width = modelWidth;
      canvas.height = modelHeight;
      const ctx = canvas.getContext("2d");
      
      if (!ctx) {
        reject(new Error("Could not get canvas context"));
        return;
      }

      // Save original dimensions for later scaling
      const originalWidth = img.naturalWidth;
      const originalHeight = img.naturalHeight;
      
      // Draw the image on the canvas with the model dimensions
      ctx.drawImage(img, 0, 0, modelWidth, modelHeight);
      
      // Get the image data from the canvas
      const imageData = ctx.getImageData(0, 0, modelWidth, modelHeight);
      const { data } = imageData;
      
      // Convert to RGB and normalize to [0, 1]
      const tensor = new Float32Array(3 * modelHeight * modelWidth);
      
      // YOLOv8 expects CHW (Channel, Height, Width) format
      for (let y = 0; y < modelHeight; y++) {
        for (let x = 0; x < modelWidth; x++) {
          const pixelIndex = (y * modelWidth + x) * 4;
          
          // Normalize to [0,1] and convert from RGBA to RGB
          const r = data[pixelIndex] / 255.0;
          const g = data[pixelIndex + 1] / 255.0;
          const b = data[pixelIndex + 2] / 255.0;
          
          // CHW format for each channel
          tensor[0 * modelHeight * modelWidth + y * modelWidth + x] = r;
          tensor[1 * modelHeight * modelWidth + y * modelWidth + x] = g;
          tensor[2 * modelHeight * modelWidth + y * modelWidth + x] = b;
        }
      }

      resolve({
        tensor,
        width: modelWidth,
        height: modelHeight,
        originalWidth,
        originalHeight
      });
    };

    img.onerror = () => {
      reject(new Error("Failed to load image"));
    };

    img.src = imageSrc;
  });
};

/**
 * Draws detection boxes on a canvas
 */
export const drawDetections = (
  canvas: HTMLCanvasElement,
  detections: {
    bbox: [number, number, number, number];
    score: number;
    class_id: number;
    class_name: string;
  }[],
  threshold: number = 0.25
): void => {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw the image again to show the detections on top
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Draw bounding boxes for each detection above threshold
    detections
      .filter(detection => detection.score >= threshold)
      .forEach(detection => {
        const [x, y, width, height] = detection.bbox;
        
        // Draw bounding box
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#00FF00";
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.stroke();
        
        // Draw label background
        const label = `${detection.class_name}: ${Math.round(detection.score * 100)}%`;
        const textMetrics = ctx.measureText(label);
        ctx.fillStyle = "rgba(0, 255, 0, 0.7)";
        ctx.fillRect(
          x, 
          y - 25, 
          textMetrics.width + 10, 
          25
        );
        
        // Draw label text
        ctx.fillStyle = "#000000";
        ctx.font = "16px Arial";
        ctx.fillText(label, x + 5, y - 8);
      });
  };
  
  // Set source to canvas.toDataURL() to ensure we draw on top of the current state
  img.src = canvas.toDataURL();
};
