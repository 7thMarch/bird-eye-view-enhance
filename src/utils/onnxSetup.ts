
import * as ort from 'onnxruntime-web';

/**
 * Configure ONNX Runtime with optimal settings
 */
export function configureOnnxRuntime() {
  // Set WebAssembly thread count based on hardware
  const numThreads = Math.min(
    navigator.hardwareConcurrency || 4, 
    4  // Cap at 4 threads to avoid excessive resource usage
  );
  
  ort.env.wasm.numThreads = numThreads;

  // Try to use WebGPU if available
  const executionProviders = ['wasm'];
  
  // Check for WebGPU support
  // Using 'any' type assertion to avoid TypeScript error since 'gpu' property is not in standard Navigator type
  if (
    typeof navigator !== 'undefined' && 
    'gpu' in navigator && 
    ort.env.webgpu !== undefined
  ) {
    executionProviders.unshift('webgpu');
  } else if (ort.env.webgl !== undefined) {
    // Fall back to WebGL if WebGPU is not available
    executionProviders.unshift('webgl');
  }

  console.log(`ONNX Runtime configured with ${numThreads} threads and providers: ${executionProviders.join(', ')}`);

  return {
    executionProviders,
    graphOptimizationLevel: 'all' as const,
    numThreads
  };
}

/**
 * Get the best available backend for ONNX Runtime
 */
export function getBestAvailableBackend(): string {
  if (
    typeof navigator !== 'undefined' && 
    'gpu' in navigator && 
    ort.env.webgpu !== undefined
  ) {
    return 'webgpu';
  } else if (ort.env.webgl !== undefined) {
    return 'webgl';
  }
  return 'wasm';
}

/**
 * Check if the given backend is supported
 */
export function isBackendSupported(backend: string): boolean {
  switch (backend) {
    case 'webgpu':
      return typeof navigator !== 'undefined' && 
             'gpu' in navigator && 
             ort.env.webgpu !== undefined;
    case 'webgl':
      return ort.env.webgl !== undefined;
    case 'wasm':
      return true; // WASM is always supported as a fallback
    default:
      return false;
  }
}
