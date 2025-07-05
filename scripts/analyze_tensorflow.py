#!/usr/bin/env python3
"""
TensorFlow Model Analysis Script
Analyzes TensorFlow models and outputs JSON results for PhantomGPU
"""

import sys
import json
import os
from pathlib import Path
import argparse

def analyze_model(model_path):
    """Analyze a TensorFlow model and return analysis results."""
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
    except ImportError:
        return create_fallback_analysis(model_path)
    
    model_path = Path(model_path)
    
    # Determine model format and load accordingly
    if model_path.is_dir():
        # SavedModel format
        return analyze_savedmodel(model_path)
    elif model_path.suffix == '.pb':
        # Frozen graph format
        return analyze_frozen_graph(model_path)
    elif model_path.suffix == '.h5':
        # Keras format
        return analyze_keras_model(model_path)
    elif model_path.suffix == '.tflite':
        # TensorFlow Lite format
        return analyze_tflite_model(model_path)
    else:
        return create_fallback_analysis(model_path)

def analyze_savedmodel(model_path):
    """Analyze a SavedModel format model."""
    import tensorflow as tf
    
    try:
        # Load the SavedModel
        model = tf.saved_model.load(str(model_path))
        
        # Get model signatures
        signatures = model.signatures
        
        # Get the default signature or first available
        if 'serving_default' in signatures:
            signature = signatures['serving_default']
        else:
            signature = list(signatures.values())[0] if signatures else None
        
        # Extract input/output information
        input_shapes = []
        output_shapes = []
        
        if signature:
            # Get input shapes
            for input_name, input_spec in signature.inputs.items():
                shape = input_spec.shape.as_list()
                input_shapes.append(shape)
            
            # Get output shapes
            for output_name, output_spec in signature.outputs.items():
                shape = output_spec.shape.as_list()
                output_shapes.append(shape)
        
        # Try to get more detailed information if it's a Keras model
        keras_model = None
        try:
            keras_model = tf.keras.models.load_model(str(model_path))
        except:
            pass
        
        if keras_model:
            return analyze_keras_model_object(keras_model, model_path, "SavedModel")
        
        # Basic analysis for non-Keras SavedModel
        total_params = estimate_parameters(input_shapes, output_shapes)
        model_size = get_model_size(model_path)
        
        return {
            "model_name": model_path.name,
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "estimated_memory_mb": (total_params * 4) / (1024 * 1024),
            "operations": ["SavedModel"],
            "layers": [],
            "model_size_mb": model_size,
            "format": "SavedModel"
        }
        
    except Exception as e:
        print(f"Error analyzing SavedModel: {e}", file=sys.stderr)
        return create_fallback_analysis(model_path)

def analyze_keras_model(model_path):
    """Analyze a Keras .h5 model."""
    import tensorflow as tf
    
    try:
        model = tf.keras.models.load_model(str(model_path))
        return analyze_keras_model_object(model, model_path, "Keras")
    except Exception as e:
        print(f"Error analyzing Keras model: {e}", file=sys.stderr)
        return create_fallback_analysis(model_path)

def analyze_keras_model_object(model, model_path, format_name):
    """Analyze a loaded Keras model object."""
    
    # Get model summary information
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    # Get input/output shapes
    input_shapes = []
    output_shapes = []
    
    # Handle different model types
    if hasattr(model, 'input_shape'):
        if isinstance(model.input_shape, list):
            input_shapes = [list(shape) for shape in model.input_shape]
        else:
            input_shapes = [list(model.input_shape)]
    
    if hasattr(model, 'output_shape'):
        if isinstance(model.output_shape, list):
            output_shapes = [list(shape) for shape in model.output_shape]
        else:
            output_shapes = [list(model.output_shape)]
    
    # Get layer information
    layers = []
    operations = set()
    
    for layer in model.layers:
        layer_info = {
            "name": layer.name,
            "operation_type": layer.__class__.__name__,
            "parameters": layer.count_params(),
            "output_shape": list(layer.output_shape) if hasattr(layer, 'output_shape') else [],
            "flops": estimate_layer_flops(layer)
        }
        layers.append(layer_info)
        operations.add(layer.__class__.__name__)
    
    model_size = get_model_size(model_path)
    
    return {
        "model_name": Path(model_path).name,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "estimated_memory_mb": (total_params * 4) / (1024 * 1024),
        "operations": list(operations),
        "layers": layers,
        "model_size_mb": model_size,
        "format": format_name
    }

def analyze_frozen_graph(model_path):
    """Analyze a frozen graph .pb file."""
    import tensorflow as tf
    
    try:
        # Load the frozen graph
        with tf.io.gfile.GFile(str(model_path), "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        # Analyze the graph
        operations = set()
        for node in graph_def.node:
            operations.add(node.op)
        
        # Estimate parameters (very rough)
        total_params = len(graph_def.node) * 1000  # Rough estimate
        model_size = get_model_size(model_path)
        
        return {
            "model_name": model_path.name,
            "input_shapes": [[1, 224, 224, 3]],  # Common default
            "output_shapes": [[1, 1000]],
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "estimated_memory_mb": (total_params * 4) / (1024 * 1024),
            "operations": list(operations),
            "layers": [],
            "model_size_mb": model_size,
            "format": "FrozenGraph"
        }
        
    except Exception as e:
        print(f"Error analyzing frozen graph: {e}", file=sys.stderr)
        return create_fallback_analysis(model_path)

def analyze_tflite_model(model_path):
    """Analyze a TensorFlow Lite model."""
    try:
        import tensorflow as tf
        
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        input_shapes = [detail['shape'].tolist() for detail in input_details]
        
        # Get output details
        output_details = interpreter.get_output_details()
        output_shapes = [detail['shape'].tolist() for detail in output_details]
        
        # Estimate parameters
        total_params = estimate_parameters(input_shapes, output_shapes)
        model_size = get_model_size(model_path)
        
        return {
            "model_name": model_path.name,
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "estimated_memory_mb": (total_params * 4) / (1024 * 1024),
            "operations": ["TFLite"],
            "layers": [],
            "model_size_mb": model_size,
            "format": "TensorFlowLite"
        }
        
    except Exception as e:
        print(f"Error analyzing TFLite model: {e}", file=sys.stderr)
        return create_fallback_analysis(model_path)

def estimate_layer_flops(layer):
    """Estimate FLOPs for a layer (rough approximation)."""
    layer_type = layer.__class__.__name__
    
    if 'Conv' in layer_type:
        # For convolutional layers
        if hasattr(layer, 'output_shape') and hasattr(layer, 'kernel_size'):
            output_shape = layer.output_shape
            kernel_size = layer.kernel_size
            if isinstance(output_shape, (list, tuple)) and len(output_shape) >= 3:
                # Rough estimate: output_elements * kernel_size * channels
                return int(output_shape[1] * output_shape[2] * output_shape[3] * 
                          kernel_size[0] * kernel_size[1])
    elif 'Dense' in layer_type:
        # For dense layers
        if hasattr(layer, 'units') and hasattr(layer, 'input_shape'):
            input_shape = layer.input_shape
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
                return int(layer.units * input_shape[-1])
    
    return 1000  # Default estimate

def estimate_parameters(input_shapes, output_shapes):
    """Estimate model parameters from input/output shapes."""
    if not input_shapes or not output_shapes:
        return 1000000  # Default
    
    input_size = sum(abs(shape[i]) for shape in input_shapes for i in range(1, len(shape)) if shape[i] != -1)
    output_size = sum(abs(shape[i]) for shape in output_shapes for i in range(1, len(shape)) if shape[i] != -1)
    
    # Simple heuristic
    return max(input_size * output_size, 1000)

def get_model_size(model_path):
    """Get model size in MB."""
    path = Path(model_path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)
    return 0.0

def create_fallback_analysis(model_path):
    """Create fallback analysis when TensorFlow is not available."""
    path = Path(model_path)
    
    if path.is_dir():
        format_name = "SavedModel"
        estimated_params = 25_000_000
    elif path.suffix == '.pb':
        format_name = "FrozenGraph"
        estimated_params = 6_000_000
    elif path.suffix == '.h5':
        format_name = "Keras"
        estimated_params = 5_000_000
    elif path.suffix == '.tflite':
        format_name = "TensorFlowLite"
        estimated_params = 1_000_000
    else:
        format_name = "Unknown"
        estimated_params = 1_000_000
    
    model_size = get_model_size(model_path)
    
    return {
        "model_name": path.name,
        "input_shapes": [[1, 224, 224, 3]],
        "output_shapes": [[1, 1000]],
        "total_parameters": estimated_params,
        "trainable_parameters": estimated_params,
        "estimated_memory_mb": (estimated_params * 4) / (1024 * 1024),
        "operations": ["Unknown"],
        "layers": [],
        "model_size_mb": model_size,
        "format": format_name
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze TensorFlow models")
    parser.add_argument("model_path", help="Path to the TensorFlow model")
    parser.add_argument("--format", choices=["json"], default="json", 
                       help="Output format (default: json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        analysis = analyze_model(args.model_path)
        print(json.dumps(analysis, indent=2))
    except Exception as e:
        print(f"Error analyzing model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 