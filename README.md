
# Medical image segmentation using TensorFlow
The new owner of this repo is huappe. This repo contains code for segmenting medical images using TensorFlow.
First, you would need to have the Python packages h5py, SimpleITK and, of course, TensorFlow.

The assumption is that you have Niftii files (.nii.gz), with all the training images being in a folder that contains the training subjects as separate folders. Each training subject will be a folder, where the name of this folder should match the name of the CT image. The ground truth should be located in the same folder and should be called GT.nii.gz - an image with each voxel having values ranging from 0 to num_classes-1.
```
Data
|
|--sub1/
    |--sub1.nii.gz
    |--GT.nii.gz
|--sub2/
    |--sub2.nii.gz
    |--GT.nii.gz
|
|...
```
The name can be different as long as the CT file and folder name are identical.