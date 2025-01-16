### Enhanced README for Data Importing Module

---

# **Data Importing and Preprocessing Module**

This module processes satellite imagery and label data to create datasets suitable for training deep learning-based segmentation models. It includes data normalization, patch extraction, train/test data separation, label-based filtering, missing value imputation, and dynamic min-max normalization. The processed dataset is saved in `.pkl` format for efficient reuse.

---

## **Features**
### **1. Data Preprocessing**
- Satellite imagery is normalized to scale pixel values within `[0, 1]`.
- **Min-Max Normalization**: Dynamic minimum and maximum values are calculated for each satellite orbits (`ASC`, `DSC`) to adjust the input data.
- Label data is structured with **background (0)** and **water body (1)** classes for segmentation tasks.

### **2. Min-Max Value Calculation**
- For each season (`ASC`, `DSC`), the minimum and maximum pixel values of the satellite imagery are computed dynamically:
  - Excludes missing values (`-9999`) to ensure accurate normalization.
  - These calculated values are used to normalize the input data to a consistent scale.


### **3. Coordinate-Based Train/Test Data Splitting**
- Patches are split into train and test datasets based on x-coordinates to prevent overlap:
  - **x-coordinate ≤ 8000**: Train data
  - **x-coordinate > 8000**: Test data
- This method ensures temporal data consistency, improving the reliability of model evaluation.

### **4. Label-Based Data Filtering**
- Patches are filtered based on the proportion of **water body (1)** labels:
  - Only patches where the water body pixels make up at least 1% of the total patch size are retained.

### **5. Patch Generation**
- Images are divided into patches of size `512x512` with a 25% overlap.
- Patches containing missing label values are automatically excluded.

### **6. Missing Value Handling**
- Missing values (`-9999`) are imputed using a **KNN Imputer**:
  - Nearest neighbors (`n_neighbors=5`) are used to replace missing values with interpolated data.


### **7. Data Saving**
- Processed datasets are saved in `.pkl` format for efficient loading and reuse.

---


## **Dataset Structure**
Ensure the dataset follows this directory structure:
```
my_path/
└── TSX/
    ├── image/
    │   ├── ASC/
    │   └── DSC/
    └── Label/
        ├── ASC/
        └── DSC/
```

### **Data Format**
1. **Image Data (`image/`)**
   - GeoTIFF files named in the format: `YYYYMMDD_TSX.tif`.
   - Each file contains satellite observation data.
   - Data type : Float32

2. **Label Data (`Label/`)**
   - GeoTIFF files named in the format: `YYYYMMDD_TSX_GRID_label.tif`.
   - **0**: Background  
   - **1**: Water Body  

---

## **Usage**
### **1. Key Parameters**
- **`my_path`**: Specifies the base directory of the dataset.
- **Patch Size**: 512x512
- **Overlap Ratio**: 25%
- **Label Threshold**: Minimum 1% of water body (1) pixels per patch.


### **2. Outputs**
- Processed data is saved in the following format:
  ```
  {my_path}/TSX_ai_data.pkl
  ```
- Data structure:
  - **Train Data**
    - `ai_data["TSX"]["train"]["image"]`: Train image patches
    - `ai_data["TSX"]["train"]["label"]`: Train label patches
  - **Test Data**
    - `ai_data["TSX"]["test"]["image"]`: Test image patches
    - `ai_data["TSX"]["test"]["label"]`: Test label patches

---

## **Data Preprocessing Workflow**
1. **Min-Max Value Calculation**
   - Minimum and maximum pixel values are computed for each satellite orbit (`ASC`, `DSC`).
 
2. **Normalization**
   - Input data is normalized using the calculated min and max values.
   - Normalized values fall within the range `[0, 1]`.

3. **Patch Generation**
   - Images are divided into patches of size `512x512` with 25% overlap.
   - Patches containing missing label values are excluded.

4. **Coordinate-Based Data Splitting**
   - Patches are split into train and test datasets based on their x-coordinates:
     - **x ≤ 8000**: Train data
     - **x > 8000**: Test data

5. **Label-Based Filtering**
   - Patches are filtered based on the proportion of water body (1) labels

6. **Handling Missing Values**
   - Missing values are imputed using the **KNN Imputer**:

7. **Saving Processed Data**
   - Final datasets are saved in `.pkl` format for reuse.

---

## **Notes**
1. Min-max normalization ensures consistent scaling of input data for each satellite orbits, improving model performance.
2. Coordinate-based splitting prevents overlap between train and test datasets, maintaining data integrity in time-series analysis.
3. Label filtering ensures sufficient representation of water body labels, enhancing data quality.
4. Missing value imputation minimizes data loss and improves model training stability.

---

## Figures
Figure 1: TerraSAR-X Intensity image (Input data)
Figure 2: TerraSAR-X Water Body image (Label data)
Figure 3: TerraSAR-X  AI data (Input & Labeld data)

