# DeepFaintEdges

For Keras:
Demo of detection of faint edges in noisy images. A python notebook that applies the edge detection CNN on a real image as described in the ICPR2020 paper "Multi-scale Processing of Noisy Images using Edge Preservation Losses": https://ieeexplore.ieee.org/document/9413325. 
The supplied h5 checkpoint is a result of training a U-net to detect edges at low SNR given a clean binary image dataset.

For pytorch:
Source Folder: "DeepEdges"
Train: train_unet.py
Test: test_unet.py

The algorihtm is trained on the binary image dataset of:
http://www.imageprocessingplace.com/root_files_V3/image_databases.htm

Good luck!
