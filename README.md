# 3d-face-preprocessing
This is a repo for 3d face preprocessing based on insightFace, opencv and open3D.

# files
## preprocess pipeline
preprocess_lock3d.py: a python implementation of the whole LED3D prprocessing pipeline incoporate with hole-filling based on open3D
## hole-filling based on open3D
open3d_fill_holes.py: a demo of hole-filling based on open3D mesh
## hole boundary detection
open3d_boundary_detection.py: a demo of open3D Boundary-detection which is used to detect hole boundary.

# Usage
release Lock3D.zip and run preprocess_lock3d.py

# requirements
Cython>=0.29.28
cmake>=3.22.3    
numpy>=1.22.3
open3d
insightFace==0.7.3
opencv-python
