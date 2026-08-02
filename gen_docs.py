import os

def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except:
        return f"Error reading file: {path}"

# Paths
app_py = 'app.py'
models_py = 'models.py'
mssa_layers_py = 'mssa_layers.py'
train_py = '../latest_data/train.py'

content = f"""# Dehradun Waste Watch: Comprehensive Technical Documentation

## 1. Project Overview
**Dehradun Waste Watch** is an end-to-end AI system designed to modernize urban waste management. It consists of a sophisticated computer vision pipeline (MSSA-YOLOv8) and a robust web application built with FastAPI. The project addresses the "small object detection" challenge in waste management, identifying micro-waste like cigarette butts alongside macro-waste like plastic bottles.

---

## 2. File-by-File Deep Dive

### 2.1 app.py (The Orchestrator)
This is the main entry point of the web application. It handles the API routes, user authentication, and the complex inference pipeline.

**What it does:**
- **Inference Pipeline**: Instead of a simple model call, it uses a multi-stage process:
    1.  **SAHI (Slicing Aided Hyper Inference)**: High-res images are sliced into patches to detect tiny objects.
    2.  **MSSA-YOLOv8**: The custom model processes these patches.
    3.  **COCO Gating**: A second YOLO model (trained on COCO) identifies people, cars, and animals. If a waste detection overlaps with a person (e.g., someone holding a bottle), it is suppressed to avoid false reports.
- **Authority Mapping**: It contains a hardcoded mapping of Dehradun localities to specific Nagar Nigam wards and representatives.
- **WebSocket Manager**: Tracks active connections to broadcast new reports to the admin dashboard in real-time.
- **Auth System**: Implements JWT (JSON Web Tokens) for secure citizen and authority login.

### 2.2 mssa_layers.py (The Innovation)
This file defines the Multi-Scale Spatial Attention (MSSA) module, the core research contribution of the project.

**What it does:**
- **MSSA Module**: Captures features at three different scales using **Dilated Convolutions** (dilation rates 1, 2, and 4). This allows the model to see the "texture" of small waste and the "context" of large waste simultaneously.
- **Attention Mechanisms**:
    - **Channel Attention**: Focuses on "what" is in the image by recalibrating feature maps.
    - **Spatial Attention**: Focuses on "where" the objects are.
- **Dynamic Injection**: The `inject_mssa` function allows us to modify a standard YOLOv8 model at runtime without changing the original library code.

### 2.3 models.py (The Foundation)
Defines the database structure using SQLAlchemy.

**What it does:**
- **User Table**: Stores user credentials, roles (Citizen, Authority, Admin), and organization details.
- **WasteReport Table**: The heart of the data. It tracks the entire lifecycle:
    - `Reported` (Initial state)
    - `Taken Up` (Authority acknowledged)
    - `Being Solved` (Cleaning in progress)
    - `Solved` (Resolution photo uploaded)
- **Relationships**: Uses foreign keys to link reports to the citizens who reported them and the authorities assigned to them.

### 2.4 train.py (The Brain)
Contains the logic for training the custom MSSA-YOLOv8 model.

**What it does:**
- **Data Augmentation**: Uses a custom **Copy-Paste** strategy where micro-waste objects (like cigarette butts) are synthetically pasted onto random backgrounds to help the model learn rare classes.
- **Hybrid Loss**: Combines **Focal Loss** (to handle class imbalance) and **CIoU Loss** (for precise bounding box regression).
- **HPC Optimization**: Includes logic to detect H100 GPUs and optimize Tensor Core usage for faster training.

---

## 3. Key Algorithms & Concepts

### 3.1 SAHI (Slicing Aided Hyper Inference)
Standard YOLO models resize high-res images to 640x640, making small objects (like a 20px cigarette butt) disappear. SAHI slices the 4K/1080p image into 512x512 overlapping windows, runs inference on each, and merges the results using Non-Maximum Suppression (NMS).

### 3.2 MSSA (Multi-Scale Spatial Attention)
Standard convolutions have a fixed receptive field. By using parallel dilated convolutions, MSSA gathers information from a wider area without losing resolution. The attention maps then "highlight" the pixels most likely to contain waste.

### 3.3 COCO Gating
A significant problem in waste detection is "Context Errors" (e.g., a bottle in a shop window being flagged as litter). COCO Gating suppresses waste detections that have high IoU (Intersection over Union) with non-waste COCO categories like `person`, `car`, or `dining table`.

---

## 4. Anticipated Presentation Questions (Q&A)

### 4.1 Concept Questions
1.  **Q: Why use Dilated Convolutions in the MSSA module?**
    *   **A**: Dilated convolutions increase the receptive field without increasing the number of parameters or losing spatial resolution. This allows the model to understand the global context (where the waste is) while maintaining a high-resolution view of small details.
2.  **Q: What is the benefit of Focal Loss over Standard Cross Entropy?**
    *   **A**: Standard Cross Entropy is dominated by "easy" majority classes. Focal Loss adds a modulating factor `(1-pt)^gamma` that reduces the loss contributed by easy examples and focuses the training on "hard" examples (like tiny, blurry waste objects).
3.  **Q: How does the system handle "false positives" in a crowded street?**
    *   **A**: Through COCO Gating. If a "plastic bottle" is detected, but it overlaps significantly with a "person" detected by the COCO model, the system assumes the person is carrying the bottle and suppresses the alert.

### 4.2 Code-Specific Questions
1.  **Q: In `app.py`, how is a report assigned to an authority?**
    *   **A**: Inside the `create_report` endpoint, we use the `locality` string from the form data. We then query the `User` table for a user with `role="authority"` whose `assigned_locality` matches the report's locality.
2.  **Q: How do you "inject" the MSSA layers into the pre-trained YOLO model?**
    *   **A**: In `mssa_layers.py`, the `inject_mssa` function iterates through the model's sequential layers. When it finds a `C2f` block (at index 11 or higher in the neck), it replaces it with a `nn.Sequential` block containing the original `C2f` layer followed by our new `MSSA` module.
3.  **Q: Where is the "After Photo" stored when an authority marks a task as solved?**
    *   **A**: It is handled in the `/api/authority/reports/{{report_id}}/update` endpoint. The image is saved to the `static/uploads` directory with a `solved_` prefix, and the path is stored in the `after_image_path` column of the `WasteReport` table.

---

## 5. Source Code Annex

### app.py
```python
{read_file(app_py)}
```

### models.py
```python
{read_file(models_py)}
```

### mssa_layers.py
```python
{read_file(mssa_layers_py)}
```

### train.py
```python
{read_file(train_py)}
```
"""

with open('detailed_project_documentation.md', 'w') as f:
    f.write(content)

print("Documentation generated successfully.")
