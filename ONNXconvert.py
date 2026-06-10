import torch
import onnx
import os
from model import XAI_CTI_Model

def convert_to_onnx():
    """Convert PyTorch model to ONNX format with validation."""
    
    try:
        print("=" * 60)
        print("PyTorch to ONNX Conversion Script")
        print("=" * 60)
        
        # Check if model file exists
        if not os.path.exists('xai_cti_model.pth'):
            raise FileNotFoundError("xai_cti_model.pth not found in current directory")
        
        print("\n[1/4] Loading PyTorch model...")
        model = XAI_CTI_Model(input_dim=73)
        model.load_state_dict(torch.load('xai_cti_model.pth'))
        model.eval()
        print("✓ Model loaded successfully")
        
        # Create dummy input matching model's expected input
        print("\n[2/4] Creating dummy input for conversion...")
        dummy_input = torch.randn(1, 73)
        print("✓ Dummy input created: shape", dummy_input.shape)
        
        # Convert to ONNX
        print("\n[3/4] Converting to ONNX format...")
        torch.onnx.export(
            model,
            dummy_input,
            'xai_cti_model.onnx',
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
            do_constant_folding=True,
            verbose=False
        )
        print("✓ ONNX export completed")
        
        # Validate ONNX model
        print("\n[4/4] Validating ONNX model...")
        onnx_model = onnx.load('xai_cti_model.onnx')
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
        
        # Print model info
        file_size = os.path.getsize('xai_cti_model.onnx') / (1024 * 1024)
        print("\n" + "=" * 60)
        print("Conversion Successful!")
        print("=" * 60)
        print(f"Output file: xai_cti_model.onnx")
        print(f"File size: {file_size:.2f} MB")
        print(f"Input name: input")
        print(f"Output name: output")
        print("=" * 60 + "\n")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise

if __name__ == "__main__":
    convert_to_onnx()