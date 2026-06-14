import sys
import os
import torch

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dataset import get_transforms
    from models import get_model
    print("SUCCESS: Imported dataset and model modules successfully!")
except Exception as e:
    print(f"FAILED: Import error: {e}")
    sys.exit(1)

def run_tests():
    print("="*60)
    print("Running Pipeline Tests with Dummy Data...")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")
    
    # Create random batch of size 2, 3 channels, size 128x128
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    
    models_to_test = ["vanilla", "resnet", "inception", "mobilenet"]
    
    for name in models_to_test:
        print(f"\nTesting Model: {name.upper()}")
        try:
            # Load model
            model = get_model(name, pretrained=False).to(device)
            
            # Forward pass
            outputs = model(dummy_input)
            print(f"  [+] Forward Pass: Output shape {list(outputs.shape)}")
            assert outputs.shape == (2, 2), f"Expected output shape (2, 2), got {outputs.shape}"
            
            print(f"SUCCESS: Model {name.upper()} passed forward pass check!")
            
        except Exception as e:
            print(f"FAILED: Model {name.upper()} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    print("\n" + "="*60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    run_tests()
