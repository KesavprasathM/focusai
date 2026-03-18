"""
TruthLens v4.0 — Model Export Utility
Exports trained model to ONNX format for fast production inference.
3x faster than PyTorch on CPU.

Usage:
  python export_model.py
"""

import torch
import torch.onnx
from pathlib import Path
import sys
sys.path.insert(0, '.')
from app import TruthLensV4

def export_onnx():
    print("\n" + "="*50)
    print("  TruthLens — Model Export to ONNX")
    print("="*50)

    model_path = Path('models/truthlens_v4.pth')
    onnx_path  = Path('exports/truthlens_v4.onnx')

    if not model_path.exists():
        print("❌ No trained model found at models/truthlens_v4.pth")
        print("   Run python train.py first!")
        return

    # Load model
    print("\nLoading model...")
    model = TruthLensV4(num_classes=2)
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("✅ Model loaded")

    # Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 380, 380)
    Path('exports').mkdir(exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image':  {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=17,
        verbose=False
    )

    size_mb = onnx_path.stat().st_size / (1024*1024)
    print(f"✅ Exported to {onnx_path}")
    print(f"   File size: {size_mb:.1f} MB")

    # Export quantized (smaller + faster)
    print("\nExporting quantized model...")
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    quant_path = Path('exports/truthlens_v4_quantized.pth')
    torch.save(quantized.state_dict(), quant_path)
    size_q = quant_path.stat().st_size / (1024*1024)
    print(f"✅ Quantized model saved to {quant_path}")
    print(f"   File size: {size_q:.1f} MB")

    print("\n" + "="*50)
    print("  Export Complete!")
    print(f"  ONNX model    : {onnx_path}")
    print(f"  Quantized     : {quant_path}")
    print("  Use ONNX for 3x faster inference in production")
    print("="*50 + "\n")


if __name__ == '__main__':
    export_onnx()