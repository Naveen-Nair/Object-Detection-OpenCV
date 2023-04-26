# Introduction:
This project is an implementation of object detection using contour detection and feature matching using C++ and OpenCV. It detects objects in an image and draws bounding boxes around them. The project uses contour detection to detect edges in the image, and feature matching techniques to match the edges with feature vectors. It then computes the bounding boxes of the detected objects and overlays them on the original image. We have further extended it to perform its functions in video as well

## Steps involved in Object Detection:
The project involves the following steps:

1. Load and Preprocess the Image: Load the image using OpenCV and preprocess it by converting it to grayscale, applying thresholding, and removing noise using Gaussian blur, also use gradient mapping using sobel operator
2. Detect Edges using Contour Detection: Use the preprocessed image to detect edges using contour detection. This will generate a list of contours that represent the edges of the objects present in the image.
3. Create Feature Vectors: For each contour detected, extract a set of feature vectors that can be used to represent the contour. This is done by fourier transforming the vector, and converting into fourier descriptors.
4. Build kd Tree: Build a kd tree using the feature vectors generated in step 3. This will enable efficient search for matching feature vectors in the next step.
5. Search for Matching Feature Vectors: For each contour detected in the original image, extract the corresponding feature vectors and search for matching feature vectors in the kd tree built in step 4, using KNN algorithm.
6. Compute Object Bounding Boxes: For each matching feature vector found in step 5, compute the bounding box of the corresponding contour in the original image.
7. Visualize the Results: Finally, visualize the original image with the bounding boxes of the detected objects overlaid on top.

## Dependencies:
To run this project, you need to install OpenCV and its dependencies. You can install them by following the instructions provided on the OpenCV website.

## Instructions to Run:
Once you have installed OpenCV and its dependencies, you can run this project using the following steps:

1. To check the image processing functionality, go to the 'imageProcessing' directory and compile the files using the following command:
``` 
g++ -std=c++17 -ggdb pkg-config --cflags --libs opencv4 imageprocessing.cpp 1.preprocessing.cpp 2.edgeDetection.cpp 3.extractFeatures.cpp 4.buildKDtree.cpp 5.matchFeatures.cpp 6.boundingBoxes.cpp -o a.out 
```
Run the compiled file using the following command:
```
./a.out
```
This will apply the object detection algorithm to the sample image in the directory and display the result in the processing folder.

2. To check the video processing functionality, go to the 'videoProcessing' directory and compile the files using the following command:
``` 
g++ -std=c++17 -ggdb pkg-config --cflags --libs opencv4 videoProcessing.cpp 1.preprocessing.cpp 2.edgeDetection.cpp 3.extractFeatures.cpp 4.buildKDtree.cpp 5.matchFeatures.cpp 6.boundingBoxes.cpp -o a.out 
```
Run the compiled file using the following command:
```
./a.out
```
This will apply the object detection algorithm to the sample video in the directory and display the result.

### Conclusion:
Object detection is a widely used computer vision application that has numerous practical applications. This project implements a simple yet effective method for detecting objects in an image. With some modifications, it can be applied to videos as well. 