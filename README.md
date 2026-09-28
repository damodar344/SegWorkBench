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

## Citation

If you use SegWorkbench in your research, please cite our paper:

```bibtex
@inproceedings{dhital2026segworkbench,
  title={SegWorkbench: A Unified Framework for Deep Learning-Based and Interactive Image Segmentation},
  author={Dhital, Damodar and Pokhrel, Abishek and Ashu, Favour and Darm, Vir Chuy and Qiao, Guanda and Zhang, Lei},
  booktitle={2026 IEEE/ACIS 24th International Conference on Software Engineering Research, Management and Applications (SERA)},
  pages={231--236},
  year={2026},
  publisher={IEEE},
  doi={10.1109/SERA69989.2026.11618632}
}

Authors:
Damodar Dhital
Towson University

Advisor: Dr. Lei Zhang
