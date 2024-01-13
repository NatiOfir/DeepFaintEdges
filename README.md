# DeepFaintEdges

Demo of detection of faint edges in noisy images. 
The code is based on the edge detection CNN applied on a
real or binary image as described in the ICPR2020 paper 
"Multi-scale Processing of Noisy Images using Edge Preservation Losses": 
https://ieeexplore.ieee.org/document/9413325.


For Keras:

1) demo.ipynb is a python notebook that applies the edge detection on a real images

2) The supplied h5 checkpoint is a result of training a U-net to detect edges
at low SNR given a clean binary image dataset.

For PyTorch: (If you want to train, test and edit source code)

1) pip install -r DeepEdges/requirements.txt

2) Add source folder: "DeepEdges" (mark directory as source root)

3) Train: train_unet.py

4) Test: test_unet.py (binary images)
5) Test: test_real_image_unet.py (natural images)


The algorithm is trained on the binary image dataset of:
http://www.imageprocessingplace.com/root_files_V3/image_databases.htm

Good luck!
