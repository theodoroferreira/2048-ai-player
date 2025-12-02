import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import tf2onnx
import onnx

def convert_model(
    keras_model_path: str = "python/models/model_1000.h5",
    output_path: str = "model/model_1000.onnx"
):
    if not os.path.exists(keras_model_path):
        print(f"Error: Model file not found at {keras_model_path}")
        print("Please train the model first using train_dqn.py")
        sys.exit(1)

    print(f"Loading Keras model from {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path, compile=False)

    print(f"Model loaded successfully!")
    print(f"Model summary:")
    model.summary()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nConverting to ONNX format...")

    import shutil
    import subprocess

    temp_saved_model_path = "temp_saved_model"
    print(f"Saving to temporary SavedModel format...")
    model.export(temp_saved_model_path)

    print(f"Converting SavedModel to ONNX...")
    result = subprocess.run([
        "python", "-m", "tf2onnx.convert",
        "--saved-model", temp_saved_model_path,
        "--output", output_path,
        "--opset", "13"
    ], capture_output=True, text=True)

    shutil.rmtree(temp_saved_model_path, ignore_errors=True)

    if result.returncode != 0:
        print(f"Error during conversion:")
        print(result.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Conversion successful!")
    print(f"ONNX model saved to: {output_path}")
    print(f"\nModel details:")
    print(f"  - Input shape: (batch_size, 16)")
    print(f"  - Output shape: (batch_size, 4)")
    print(f"  - ONNX opset: 13")
    print(f"{'='*60}")
    print(f"\nYou can now use this model in the browser with ONNX Runtime Web!")
    print(f"See: https://onnxruntime.ai/docs/tutorials/web/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Keras model to ONNX")
    parser.add_argument(
        "--input",
        default="python/models/model_1000.h5",
        help="Path to input Keras model (.h5 file)"
    )
    parser.add_argument(
        "--output",
        default="model/model_1000.onnx",
        help="Output path for ONNX model"
    )

    args = parser.parse_args()

    convert_model(args.input, args.output)
