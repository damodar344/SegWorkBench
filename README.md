SegWorkbench

-U-Net training
-Batch inference
-GMM/Otsu/IsoData thresholding
-Interactive mask editing

The framework was developed as part of the research paper:

"SegWorkbench: A Unified Framework for DeepLearning-Based and Interactive Image Segmentation"

Requirements
!pip install torch torchvision numpy matplotlib pillow scikit-learn scikit-image

Run the Application

The main application entry point is:

Main app.py

Launch the GUI using:

python main.py
Dataset Structure
images/
masks/

Each image should have a corresponding mask with the same filename.

Features
-U-Net model training
-Batch segmentation inference
-Statistical thresholding (GMM, Otsu, IsoData)
-Manual mask refinement
-Training loss visualization

Authors:
Damodar Dhital
Towson University

Advisor: Dr. Lei Zhang
