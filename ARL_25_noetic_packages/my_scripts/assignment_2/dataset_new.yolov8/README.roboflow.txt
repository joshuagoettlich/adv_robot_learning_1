
RL_Dataset_cleanup - v3 Added Augmentation
==============================

This dataset was exported via roboflow.com on June 16, 2025 at 10:07 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 638 images.
Cubes are annotated in YOLOv8 format.

The following pre-processing was applied to each image:

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random brigthness adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 4 pixels


