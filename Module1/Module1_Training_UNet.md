### README for UNet Training Module

---

# **UNet Training Module**

This module implements the training pipeline for a UNet-based model designed for semantic segmentation tasks using satellite imagery. It includes data preprocessing, model training, and performance evaluation. The module utilizes TensorFlow and Keras to build and train the UNet model.

---

## **Features**
### **1. Data Load**
- **Dataset**: The training and testing datasets are loaded from a `.pkl` file.

### **2. Model Architecture**
- **UNet**: A convolutional neural network (CNN) for semantic segmentation.
- **Input Size**: Matches the shape of the training data.
- **Output**: Two-class segmentation (water and non-water).
- **Hyperparameters**:
  - Learning Rate: `0.0001`
  - Batch Size: `10`
  - Class Weights: `[0.15, 0.85]` (for imbalanced classes).

### **3. Training**
- **Loss Function**: Weighted sparse categorical cross-entropy.
- **Optimizer**: Adam.
- **Metrics**: Accuracy.
- **Early Stopping**: Stops training when validation loss doesn't improve for `1000` epochs.
- **Model Checkpointing**: Saves the best model based on validation loss.
- **Learning Rate Adjustment**: Reduces the learning rate when validation loss plateaus.

### **4. Visualization**
- Plots training and validation accuracy/loss curves.
- Displays sample training data and labels.

---

## **Installation**
### Prerequisites
- Python 3.11.0
- TensorFlow 2.14


---

## **Usage**
1. **Prepare Data**:
   Ensure the preprocessed `.pkl` dataset is available at the specified path (`my_path`).

2. **Output Files**:
   - Trained model weights: `checkpoint/TSX_Unet_Commit.h5`.
   - Training history: `TSX_Unet_commit.pickle`.

---

## **Code Overview**
### **1. Data Preprocessing**
- Loads training and testing datasets from `TSX_ai_data.pkl`.
- Normalizes input data and converts labels to `uint8`.

### **2. Model Training**
- Sets up the UNet architecture using `get_unet` from `WB_TSX`.
- Compiles the model with custom loss and metrics.
- Trains the model with early stopping, learning rate adjustment, and checkpointing.

### **3. Visualization**
- Displays random samples of training data and corresponding labels.
- Plots training and validation accuracy/loss curves.

---

### Training History Visualization
- **Accuracy Curve**: Figure 1
  ![Accuracy](path/to/accuracy_plot.png)

- **Loss Curve**:  Figure 2
  ![Loss](path/to/loss_plot.png)

---

## Figures
Figure 1: Training History Visualization (Accruacy curve)
Figure 2: Training History Visualization (Loss curve)
