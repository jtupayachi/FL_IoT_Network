#!/usr/bin/env python3
"""
Test PyTorch installation and basic functionality.
"""

import sys
import torch
import torch.nn as nn
import numpy as np

def test_pytorch_installation():
    """Run a series of sanity checks for PyTorch installation."""
    print("🧪 Testing PyTorch Installation")
    print("=" * 60)

    # --- Test basic import ---
    try:
        print("✅ PyTorch imported successfully")
        print(f"PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False

    # --- Test CUDA availability ---
    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU backend")
    except Exception as e:
        print(f"⚠️ CUDA check failed: {e}")

    # --- Test tensor operations ---
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print(f"✅ Tensor operations work: {x.tolist()} + {y.tolist()} = {z.tolist()}")
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

    # --- Test neural network forward pass ---
    try:
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        test_input = torch.randn(1, 10)
        output = model(test_input)
        print(f"✅ Neural network works: input {tuple(test_input.shape)} → output {tuple(output.shape)}")
    except Exception as e:
        print(f"❌ Neural network failed: {e}")
        return False

    # --- Test NumPy ↔ PyTorch interoperability ---
    try:
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        torch_tensor = torch.from_numpy(np_array)
        back_to_numpy = torch_tensor.numpy()
        if np.allclose(np_array, back_to_numpy):
            print("✅ NumPy integration works")
        else:
            print("⚠️ NumPy values differ after conversion")
    except Exception as e:
        print(f"❌ NumPy integration failed: {e}")
        return False

    print("=" * 60)
    print("🎉 All tests completed successfully!")
    return True


if __name__ == "__main__":
    success = test_pytorch_installation()
    sys.exit(0 if success else 1)
