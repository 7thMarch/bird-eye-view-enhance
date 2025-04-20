
export interface BirdNestDetection {
  bbox: [number, number, number, number]; // [x, y, width, height]
  score: number;
  class_id: number;
  class_name: string;
}

export interface PreprocessedImageData {
  tensor: Float32Array;
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
}
