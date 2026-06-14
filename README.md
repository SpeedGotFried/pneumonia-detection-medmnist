# Medical Image Classification: Pneumonia Detection on MedMNIST

This repository contains a PyTorch-based framework for classifying pediatric chest X-ray images as **Normal** or **Pneumonia** using the `PneumoniaMNIST` dataset from MedMNIST. It compares four deep learning architectures: **Vanilla CNN**, **ResNet-18**, **GoogLeNet (Inception)**, and **MobileNet-V2**.

---

## Part 1: Dataset & Problem Interpretation

### 1. The Clinical Problem
Pneumonia is an inflammatory condition of the lung affecting primarily the microscopic air sacs known as alveoli. It is a leading cause of mortality among children worldwide. Chest X-ray (CXR) imaging is the primary diagnostic tool. However, interpreting pediatric chest X-rays is challenging due to:
- Varied clinical presentations and overlapping patterns.
- High inter-observer variability among radiologists.
- Subjective interpretation, especially in low-resource environments.

By automating the screening process using deep learning, we can provide:
- Rapid triage support.
- Second-opinion tools for clinicians.
- Standardized, objective diagnostic metrics.

### 2. Dataset Specifics (`PneumoniaMNIST`)
- **Data Source**: A cohort of pediatric chest X-ray images (1-5 years old) from Guangzhou Women and Children’s Medical Center.
- **Grayscale Images**: Standardized and resized to $28 \times 28$ pixels.
- **Binary Classification**:
  - `0`: Normal (No infection)
  - `1`: Pneumonia (Bacterial or viral infection)
- **Data Splits**:
  - **Training Set**: 4,708 images
  - **Validation Set**: 524 images
  - **Test Set**: 624 images

### 3. Class Imbalance & Evaluation Metrics
The dataset is significantly imbalanced:
- **Pneumonia (Class 1)**: ~75% of the data.
- **Normal (Class 0)**: ~25% of the data.

**Why Accuracy is Deceptive**: If a model simply predicts "Pneumonia" for every single image, it will achieve **~75% accuracy** without learning any actual medical features. Therefore, we evaluate performance using:
1. **Accuracy**: Overall fraction of correct predictions.
2. **Precision**: Out of all predicted pneumonia cases, how many were actual pneumonia? (Reduces false positives).
3. **Sensitivity / Recall**: Out of all actual pneumonia cases, how many did the model identify? (Crucial for medical safety to minimize false negatives).
4. **Specificity**: Out of all actual normal cases, how many did the model correctly label as normal? (Reduces unnecessary treatment).
5. **F1-Score**: Harmonic mean of Precision and Recall.
6. **AUC-ROC**: Area under the ROC curve, measuring the model's ability to distinguish between classes at various threshold settings.

---

## Part 2: PyTorch Concepts & Understanding

For your meeting presentation, here are the key PyTorch building blocks implemented in this codebase:

### 1. Tensors: The Fundamental Unit
PyTorch tensors are multi-dimensional arrays (like NumPy arrays) but can be run on GPUs to accelerate computing.
- In `src/dataset.py`, transforms convert images into PyTorch float tensors of shape `[Channels, Height, Width]`, normalized to standard distributions:
  ```python
  transforms.ToTensor()
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ```

### 2. Computational Graphs & Autograd
PyTorch uses a **dynamic computation graph** constructed on-the-fly during the forward pass.
- **`loss.backward()`** computes the gradients of the loss function with respect to all learnable parameters ($w, b$) using backpropagation.
- **`optimizer.step()`** updates the parameters using these gradients (e.g., using the Adam optimizer).

### 3. Modularity with `nn.Module`
All networks inherit from `nn.Module`. This class automatically tracks learnable parameters and sub-modules.
- In `src/models.py`, `VanillaCNN` defines the architecture in `__init__` and the data flow in the `forward` function:
  ```python
  class VanillaCNN(nn.Module):
      def __init__(self):
          super().__init__()
          self.features = nn.Sequential(...)
          self.fc = nn.Linear(...)
      def forward(self, x):
          ...
  ```

### 4. Efficient Data Pipelines with `DataLoader`
`DataLoader` wraps the dataset, handles batching, shuffling, and multi-threaded data loading (using workers) to avoid CPU-to-GPU bottlenecks:
```python
DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
```

---

## Part 3: Project Structure

```
ISI Project - 2026/
├── requirements.txt      # Dependencies
├── README.md             # This document (Project documentation and background)
├── src/
│   ├── __init__.py
│   ├── dataset.py        # Loading PneumoniaMNIST, statistics, transforms
│   └── models.py         # Vanilla, ResNet, Inception, MobileNet
├── test_pipeline.py      # Local verification script
└── visualize_samples.py  # Visualizing dataset samples
```

---

## Part 4: How to Run

### Local Command Line
1. Create virtual environment and install requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate.fish  # or activate for bash/zsh
   pip install -r requirements.txt
   ```
2. Verify setup and run dummy tests:
   ```bash
   python3 test_pipeline.py
   ```
3. Run the dataset analysis:
   ```bash
   python3 src/dataset.py
   ```
4. Run sample visualization:
   ```bash
   python3 visualize_samples.py
   ```
