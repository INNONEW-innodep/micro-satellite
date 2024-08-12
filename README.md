# TerraSAR-X Data Processing for Water Body Detection

This repository contains the code and documentation for processing TerraSAR-X data to detect water bodies from raw data. The processing steps are outlined as follows:

## Overview
In this paper, we discuss the conversion of raw TerraSAR-X data into source data suitable for water body segmentation. The steps involved are:

### 1.Conversion to intensity Image
For multi-temporal TerraSAR-X Single Look Complex (SLC) data, which contains real and imaginary parts, we first convert the data to intensity images.

### 2. Look-Up Table Generation Between DEM and intensity Image
Since SLC images are in image coordinates, we use a Digital Elevation Model (DEM) for orthorectification to convert them to geographic coordinates. We simulate a DEM in image coordinates by comparing it with the geometry of the master image.

#### 2.1. Calculate the Look-Up Table (LUT) between the simulated DEM in image coordinates and the master image.

### 3. Co-Registration
To remove positional errors across different temporal images, we select the image with the highest incidence angle as the master image. We then perform co-registration of other images (slave images) to this master image.

### 4. Sigma0 Image Creation
For water body detection, we create sigma0 images, which are backscatter coefficient images. These images convert pixel values into physical properties of the Earth's surface, making them effective for land cover classification and soil moisture estimation, regardless of sensor geometry or incidence angle.

### 5. Noise Reduction
Apply a median filter to reduce noise.

### 6. Decibel Transformation
Convert the images to decibel scale to transform the Rayleigh distribution into a normal distribution.

### 7. Orthorectification using LUT
Apply the LUT created in step 2.1 to each temporal image to generate orthorectified images.

## Figures
Figure 1: TerraSAR-X SLC Image (Intensity image)
Figure 2: TerraSAR-X Orthorectified Image