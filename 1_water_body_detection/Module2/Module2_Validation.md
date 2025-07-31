### README for UNet Model Validation Module

---

# **UNet Model Validation Module**

This module validates the performance of a pre-trained UNet model for semantic segmentation tasks using test data. The module evaluates model accuracy, precision, recall, IoU, and F1-score for each class and visualizes prediction results compared to ground truth labels.

---

## **Features**
### **1. Test Data Loading**
- Loads test data from a preprocessed `.pkl` file.
- Supports satellite imagery with corresponding labels for segmentation tasks.

### **2. Model Loading and Setup**
- Loads pre-trained UNet model weights from a specified checkpoint file.
- Uses TensorFlow/Keras for inference with GPU support.

### **3. Performance Metrics**
- Calculates key metrics:
  - Accuracy
  - Intersection over Union (IoU)
  - Precision, Recall, and F1-score for each class (water and non-water).
- Generates a confusion matrix for detailed evaluation.

### **4. Visualization**
- Displays random samples of test data, predictions, and ground truth labels.
- Plots Precision-Recall curves and computes Average Precision (AP) for each class.

---

## **Installation**
### Prerequisites
- Python 3.11.0
- TensorFlow 2.14.0
---

## **Usage**
### **1. Project Structure**
Ensure the following directory structure:
```
/workspace/pytorch_unet/python-programs/Tensor/02_IITP/
├── TSX_ai_data.pkl        # Preprocessed dataset for test data
├── checkpoint/
│   └── TSX_Unet_Commit.h5 # Pre-trained model weights
```

### **2. Outputs**
- Performance metrics displayed in the terminal:
  - Accuracy, IoU, Precision, Recall, F1-score for water and non-water classes.
- Visualization:
  - Random samples of predictions compared with test data and ground truth.
  - Precision-Recall curves for each class.

---

## **Code Overview**
### **1. Data Handling**
- Loads test data (`x_test`, `t_test`) from `TSX_ai_data.pkl`.
- Normalizes the input data and converts labels to `uint8`.

### **2. Model Inference**
- Loads pre-trained UNet weights from `TSX_Unet_Commit.h5`.
- Performs inference on the test data in mini-batches for efficiency.

### **3. Performance Metrics**
- **Confusion Matrix**: Used to compute IoU, Precision, Recall, and F1-score.
- **Precision-Recall Curves**: Visualizes the model's precision-recall tradeoff for each class.
- **Overall Accuracy**: Summarizes model performance.

### **4. Visualization**
- Displays side-by-side comparisons of:
  - Predicted segmentation results.
  - Ground truth labels.
  - Input test images.

---

## **Example Validation Output**
### Performance Metrics
```
******************* Performance Results *******************
Accuracy = 0.9777

Water (Class 1) Metrics:
IoU (Water) = 0.8386
Precision (Water) = 0.8615
Recall (Water) = 0.9694
F1-score (Water) = 0.9122

Non-water (Class 0) Metrics:
IoU (Non-water) = 0.9748
Precision (Non-water) = 0.9958
Recall (Non-water) = 0.9789
F1-score (Non-water) = 0.9873
```

### Visualization
- **Results**:
  - Displays predicted masks, ground truth masks, and corresponding test images.
- **Precision-Recall Curve**:


---