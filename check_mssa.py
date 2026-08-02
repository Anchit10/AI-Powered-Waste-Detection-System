import torch
ckpt = torch.load('../mssa_waste/mssa_yolov8m/weights/best.pt', map_location='cpu', weights_only=False)
if 'model' in ckpt:
    model = ckpt['model']
    print("Model Layers:")
    for i, m in enumerate(model.model):
        print(f"Layer {i}: {type(m)}")
