#!/usr/bin/env python3
"""
TensorFlow to TensorFlow Lite Conversion Utility
Converts TensorFlow models to TFLite format for use with PhantomGPU
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} loaded successfully")
except ImportError:
    print("❌ TensorFlow not found. Install with: pip install tensorflow")
    sys.exit(1)

def convert_saved_model_to_tflite(saved_model_path: str, output_path: str = None):
    """Convert a SavedModel to TensorFlow Lite format"""
    print(f"🔄 Converting SavedModel: {saved_model_path}")
    
    if not os.path.exists(saved_model_path):
        print(f"❌ SavedModel not found: {saved_model_path}")
        return False
    
    try:
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Optional optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Determine output path
        if output_path is None:
            model_name = Path(saved_model_path).name
            output_path = f"{model_name}.tflite"
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Successfully converted to: {output_path}")
        print(f"📊 Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def convert_keras_model_to_tflite(model_path: str, output_path: str = None):
    """Convert a Keras model to TensorFlow Lite format"""
    print(f"🔄 Converting Keras model: {model_path}")
    
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Determine output path
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = f"{model_name}.tflite"
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Successfully converted to: {output_path}")
        print(f"📊 Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def convert_frozen_graph_to_tflite(pb_path: str, output_path: str = None):
    """Convert a frozen graph (.pb) to TensorFlow Lite format"""
    print(f"🔄 Converting frozen graph: {pb_path}")
    print("⚠️  Frozen graph conversion requires input/output node names")
    print("💡 Consider using SavedModel format for easier conversion")
    
    # This is more complex and requires specific input/output node names
    # For now, suggest using SavedModel format
    print("❌ Frozen graph conversion not fully implemented")
    print("💡 Convert to SavedModel first, then use this script")
    
    return False

def analyze_tflite_model(tflite_path: str):
    """Analyze a TensorFlow Lite model and show its properties"""
    print(f"🔍 Analyzing TensorFlow Lite model: {tflite_path}")
    
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📊 Model Analysis:")
        print(f"   Model size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
        print(f"   Input tensors: {len(input_details)}")
        print(f"   Output tensors: {len(output_details)}")
        
        for i, details in enumerate(input_details):
            print(f"   Input {i}: shape={details['shape']}, dtype={details['dtype'].__name__}")
        
        for i, details in enumerate(output_details):
            print(f"   Output {i}: shape={details['shape']}, dtype={details['dtype'].__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow models to TensorFlow Lite format for PhantomGPU"
    )
    parser.add_argument("input_path", help="Path to input model (SavedModel directory or .h5 file)")
    parser.add_argument("-o", "--output", help="Output .tflite file path (optional)")
    parser.add_argument("-a", "--analyze", action="store_true", help="Analyze the converted model")
    parser.add_argument("--type", choices=["saved_model", "keras", "frozen_graph"], 
                       help="Model type (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output
    
    # Auto-detect model type if not specified
    if args.type is None:
        if os.path.isdir(input_path):
            args.type = "saved_model"
        elif input_path.endswith('.h5'):
            args.type = "keras"
        elif input_path.endswith('.pb'):
            args.type = "frozen_graph"
        else:
            print("❌ Cannot auto-detect model type. Please specify with --type")
            sys.exit(1)
    
    # Convert the model
    success = False
    if args.type == "saved_model":
        success = convert_saved_model_to_tflite(input_path, output_path)
    elif args.type == "keras":
        success = convert_keras_model_to_tflite(input_path, output_path)
    elif args.type == "frozen_graph":
        success = convert_frozen_graph_to_tflite(input_path, output_path)
    
    if not success:
        print("❌ Conversion failed")
        sys.exit(1)
    
    # Analyze the converted model if requested
    if args.analyze:
        final_output = output_path if output_path else f"{Path(input_path).stem}.tflite"
        analyze_tflite_model(final_output)
    
    print("✅ Conversion complete!")
    print(f"💡 Now you can use the .tflite file with PhantomGPU:")
    print(f"   cargo run --features tensorflow -- load-model --model {final_output if output_path else Path(input_path).stem + '.tflite'} --format auto")

if __name__ == "__main__":
    main() 