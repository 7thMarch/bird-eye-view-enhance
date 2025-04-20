
import { useState, useRef, useEffect, ChangeEvent } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/custom-button';
import { preprocessImage, drawDetections } from '@/utils/imageProcessing';
import { BirdNestDetection } from '@/types';

// Define paths to static assets - these would be placed in the public folder
const MODEL_URL = '/models/birdnest.onnx'; 
const DEMO_IMAGE_URL = '/images/sample-birdnest.jpg';

const BirdNestDetector = () => {
  // State variables
  const [imageUrl, setImageUrl] = useState<string>('');
  const [detections, setDetections] = useState<BirdNestDetection[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.25);
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [workerReady, setWorkerReady] = useState<boolean>(false);
  
  // References
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  
  // Initialize worker on component mount
  useEffect(() => {
    // Create worker
    const worker = new Worker(new URL('../workers/modelWorker.ts', import.meta.url), {
      type: 'module'
    });
    
    // Set up message handler
    worker.onmessage = (event) => {
      const { type, detections, error } = event.data;
      
      switch (type) {
        case 'MODEL_LOADED':
          setIsModelLoaded(true);
          setWorkerReady(true);
          break;
        
        case 'INFERENCE_RESULT':
          if (detections) {
            setDetections(detections);
            drawDetectionsOnCanvas(detections);
          }
          setIsProcessing(false);
          break;
        
        case 'ERROR':
          setError(error || 'An unknown error occurred');
          setIsProcessing(false);
          break;
      }
    };
    
    // Store worker reference
    workerRef.current = worker;
    
    // Load model
    worker.postMessage({
      type: 'LOAD_MODEL',
      modelUrl: MODEL_URL
    });
    
    // Cleanup on unmount
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);
  
  // Handle file drops
  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpeg', '.jpg', '.webp']
    },
    onDrop: handleImageDrop,
    onDragEnter: () => setIsDragging(true),
    onDragLeave: () => setIsDragging(false)
  });
  
  // Handle file upload
  function handleImageDrop(acceptedFiles: File[]) {
    setIsDragging(false);
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const reader = new FileReader();
      
      reader.onload = (e) => {
        if (e.target && typeof e.target.result === 'string') {
          setImageUrl(e.target.result);
          setDetections([]);
        }
      };
      
      reader.readAsDataURL(file);
    }
  }
  
  // Handle example image click
  const handleExampleClick = () => {
    setImageUrl(DEMO_IMAGE_URL);
    setDetections([]);
  };
  
  // Handle detect button click
  const handleDetect = async () => {
    if (!imageUrl || !workerRef.current || !workerReady) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      // Preprocess image
      const imageData = await preprocessImage(imageUrl);
      
      // Run inference in worker
      workerRef.current.postMessage({
        type: 'RUN_INFERENCE',
        imageData,
        confidenceThreshold
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image');
      setIsProcessing(false);
    }
  };
  
  // Draw detections on canvas
  const drawDetectionsOnCanvas = (detections: BirdNestDetection[]) => {
    if (!canvasRef.current || !imageRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions to match the image
    canvas.width = imageRef.current.naturalWidth;
    canvas.height = imageRef.current.naturalHeight;
    
    // Draw the image on the canvas
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
    
    // Draw bounding boxes
    drawDetections(canvas, detections, confidenceThreshold);
  };
  
  // Handle confidence threshold change
  const handleThresholdChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    setConfidenceThreshold(value);
    
    // Redraw with new threshold if we have detections
    if (detections.length > 0) {
      drawDetectionsOnCanvas(detections);
    }
  };
  
  // Handle image load
  const handleImageLoad = () => {
    if (canvasRef.current && imageRef.current) {
      const canvas = canvasRef.current;
      
      // Set canvas size to match image
      canvas.width = imageRef.current.naturalWidth;
      canvas.height = imageRef.current.naturalHeight;
      
      // Draw image on canvas
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
      }
      
      // Redraw detections if any
      if (detections.length > 0) {
        drawDetectionsOnCanvas(detections);
      }
    }
  };
  
  // Handle saving the result
  const handleSaveResult = () => {
    if (!canvasRef.current) return;
    
    // Create a download link
    const link = document.createElement('a');
    link.download = 'birdnest-detection.png';
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-4 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Bird Nest Detector</h1>
        <p className="text-gray-600">
          Upload an image to detect bird nests using YOLOv8 directly in your browser
        </p>
      </div>
      
      {/* Model loading status */}
      {!isModelLoaded && (
        <div className="flex items-center justify-center p-4 bg-yellow-50 border border-yellow-200 rounded-md">
          <svg className="animate-spin h-5 w-5 mr-3 text-yellow-500" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p>Loading the bird nest detection model...</p>
        </div>
      )}
      
      {/* Image upload area */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
        }`}
      >
        <input {...getInputProps()} />
        {!imageUrl ? (
          <div className="space-y-2">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
            </svg>
            <p className="text-lg font-medium">Drag & drop an image here, or click to select</p>
            <p className="text-sm text-gray-500">Supports JPG, PNG, WEBP</p>
          </div>
        ) : (
          <div className="relative">
            <img
              ref={imageRef}
              src={imageUrl}
              alt="Uploaded"
              className="max-h-96 max-w-full mx-auto rounded shadow-sm"
              onLoad={handleImageLoad}
            />
            <button
              onClick={(e) => {
                e.stopPropagation();
                setImageUrl('');
                setDetections([]);
              }}
              className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        )}
      </div>
      
      {/* Example image button */}
      <div className="text-center">
        <button 
          onClick={handleExampleClick}
          className="text-blue-600 hover:text-blue-800 hover:underline"
        >
          Or try with an example image
        </button>
      </div>
      
      {/* Controls section */}
      {imageUrl && (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-4 items-center justify-between">
            <div className="space-y-1 flex-grow max-w-md">
              <label htmlFor="confidence" className="block text-sm font-medium text-gray-700">
                Confidence Threshold: {confidenceThreshold.toFixed(2)}
              </label>
              <input
                id="confidence"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={confidenceThreshold}
                onChange={handleThresholdChange}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div className="flex gap-2">
              <Button
                variant="primary"
                size="lg"
                onClick={handleDetect}
                disabled={!imageUrl || isProcessing || !isModelLoaded}
                isLoading={isProcessing}
              >
                {isProcessing ? 'Detecting...' : 'Detect Bird Nests'}
              </Button>
              
              {detections.length > 0 && (
                <Button
                  variant="outline"
                  size="lg"
                  onClick={handleSaveResult}
                >
                  Save Result
                </Button>
              )}
            </div>
          </div>
          
          {/* Error message */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded">
              <p>{error}</p>
            </div>
          )}
          
          {/* Results canvas */}
          <div className="relative bg-gray-100 rounded-lg p-2 flex justify-center">
            <canvas
              ref={canvasRef}
              className="max-w-full object-contain"
              style={{ maxHeight: '60vh' }}
            />
          </div>
          
          {/* Detection results */}
          {detections.length > 0 && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="font-semibold text-green-800 mb-2">
                Detection Results ({detections.length} bird {detections.length === 1 ? 'nest' : 'nests'} found)
              </h3>
              <ul className="space-y-1">
                {detections.map((detection, index) => (
                  <li key={index} className="text-sm text-gray-700">
                    Bird Nest #{index + 1}: Confidence {(detection.score * 100).toFixed(1)}%
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      
      {/* Tech details */}
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>Running YOLOv8 model directly in your browser using ONNX Runtime Web</p>
        <p>No data is sent to any server - all processing happens locally on your device</p>
      </div>
    </div>
  );
};

export default BirdNestDetector;
