import torch
from ultralytics import YOLO
from mssa_layers import inject_mssa
import os

# Updated to use the new model path
MODEL_PATH = 'best.pt'

if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Please run the script from the web_app directory.")
    exit(1)

print(f"Loading model from {MODEL_PATH}...")
try:
    # 1. Load the model
    model = YOLO(MODEL_PATH)
    
    # 2. Inject MSSA (if not already present in the weights)
    # Note: If weights were saved after injection, this might be redundant or needed depending on how they were saved.
    # Usually, we inject and then load.
    model = inject_mssa(model)
    
    print("\nModel Verification:")
    print(f"  Classes ({len(model.names)}): {model.names}")
    print(f"  Task: {model.task}")
    
    # Test a dummy inference
    print("\nRunning dummy inference...")
    dummy_img = torch.zeros((1, 3, 640, 640))
    results = model(dummy_img, verbose=False)
    print("  Inference successful!")
    
except Exception as e:
    print(f"\nError during model verification: {e}")
    import traceback
    traceback.print_exc()
