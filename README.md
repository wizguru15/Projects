This repository contains three distinct deep learning projects focused on industrial applications: steel surface defect classification using OpenCV, defect detection using YOLO, and predictive maintenance for turbofan engines.

-------------------------------------------
Project 1: Predictive Maintenance with LSTM
-------------------------------------------

Objective: Predict Remaining Useful Life (RUL) of turbofan engines using NASA CMAPSS dataset for predictive maintenance applications.
Dataset: NASA Turbofan Engine Degradation Simulation (FD001)

Training: 100 engines, 20,631 cycles
Test: 100 engines, 13,096 cycles
21 sensor channels + 3 operational settings
True RUL values provided for test set

Dataset Source: NASA CMAPSS Dataset | Direct Download - [https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip]

Methodology:
------------
RUL calculation with clipping at 125 cycles
MinMax scaling of sensor data
Sequence creation (50 time steps) for LSTM
3-layer LSTM architecture with dropout
Early stopping (patience=10)

Results:
--------
Mean Absolute Error (MAE): 11.54 cycles
Root Mean Squared Error (RMSE): 16.50 cycles
R² Score: 0.8423
80% of predictions accurate within 19 cycles
90% of predictions accurate within 26 cycles

Key Features:
-------------
Interactive Plotly dashboards for model evaluation
Per-engine prediction analysis
Error distribution visualization
Cumulative error analysis

-------------------------------------------------------
Project 2: Steel Defect Classification with CNN-OpenCV
-------------------------------------------------------

Objective: Classify six types of steel surface defects using a custom Convolutional Neural Network.
Dataset: NEU-DET (Northeastern University) - Contains 1,800 grayscale images of steel surfaces with six defect classes:

Crazing
Inclusion
Patches
Pitted Surface
Rolled-in Scale
Scratches

Dataset Source: NEU Surface Defect Database on Kaggle - [https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database]

Methodology:
------------
Custom OpenCV preprocessing pipeline (CLAHE, Gaussian blur, normalization)
4-block CNN architecture with BatchNormalization and Dropout
Data augmentation (rotation, shifts, zoom, flips)
Train/val/test split: 70/15/15

Results:
--------
Test Accuracy: 98.61%
Precision: 98.67%
Recall: 98.61%
F1-Score: 98.61%
Only 3 misclassifications out of 216 test samples

Key Files:
----------
preprocessing_pipeline1.png - Visualization of preprocessing steps
best_cnn_model2.h5 - Best model checkpoint
training_history2.pkl - Training metrics

-----------------------------------------------
Project 3: Steel Defect Detection with YOLOv11
-----------------------------------------------

Objective: Detect and localize steel surface defects using state-of-the-art YOLOv11 object detection.
Dataset: NEU-DET (with bounding box annotations)

Training: 1,439 images with 3,332 defect instances
Validation: 361 images with 857 defect instances
Same 6 defect classes as Project 1

Dataset Source: NEU Surface Defect Database on Kaggle (includes VOC format annotations) - [https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database]

Methodology:
------------
Converted VOC format annotations to YOLO format using Globox
YOLOv11n (nano) model architecture from Ultralytics YOLO
Trained for 80 epochs with early stopping
Image size: 200px (auto-adjusted to 224px)

-------------------------------|
Results: Overall mAP50: 71.8%  |
-------------------------------|

Per-class performance:
----------------------
Crazing: 47.3% mAP50
Inclusion: 84.4% mAP50
Patches: 92.8% mAP50 (best)
Pitted Surface: 79.9% mAP50
Rolled-in Scale: 47.3% mAP50
Scratches: 79.0% mAP50
Precision: 66.9%
Recall: 65.5%
F1-Score: 66.2%

Key Files:
----------
steel_defect_detector_2.pt - Trained YOLO model
data_fixed_2.yaml - Dataset configuration
Training logs in defect_detection_fixed_2

