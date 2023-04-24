#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>
#include <stack>

#include <iostream>

#include "1.preprocessing.hpp"
#include "2.edgeDetection.hpp"
#include "3.extractFeatures.hpp"
#include "4.buildKDtree.hpp"
#include "5.matchFeatures.hpp"
#include "6.boundingBoxes.hpp"

using namespace cv;
using namespace std;

#define KERNEL_SIZE 3
#define SIGMA 1
#define IMG_PATH "../images/disadvantage.jpeg"

#define MINBOXSIZE 1000


int main()
{
    // Load the image
    Mat image = imread(IMG_PATH);

    // Create a new grayscale image with the same size as the input image
    // CV_8UC1 - cv image with 8 bit unsigned with only one channel (ie grayscale)
    Mat gray_image(image.size(), CV_8UC1);
    grayscale(image, gray_image);
    imwrite("processing/1_gray_image_function.jpg", gray_image);

    // create a one channel image of the same size as image
    Mat thresholded_image(image.size(), CV_8UC1);
    threshold(gray_image, thresholded_image);
    imwrite("processing/2_thresholded_image.jpg", thresholded_image);

    // Apply Gaussian blur
    Mat blurred_image(image.size(), CV_8UC1);
    gaussianBlur(thresholded_image, blurred_image, KERNEL_SIZE, SIGMA);
    imwrite("processing/3_processed_image2.jpg", blurred_image);

    Mat binary_image(image.size(), CV_8UC1);
    threshold(blurred_image, binary_image);
    imwrite("processing/4_thresholded_image2.jpg", binary_image);

    Mat grad_image(image.size(), CV_8UC1);
    sobelOperator(binary_image, grad_image);

    vector<vector<Point>> contours;
    detectEdges(grad_image, contours);

    vector<vector<double>> featureVectors = extractFourierDescriptors(contours, binary_image);

    // for (int i = 0; i < featureVectors.size(); i++)
    // {
    //     cout << "Contour " << i + 1 << " Fourier descriptors: ";
    //     for (int j = 0; j < featureVectors[i].size(); j++)
    //     {
    //         cout << featureVectors[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    KDNode *root = buildKdTree(featureVectors, 0, featureVectors.size() - 1, 0);

    vector<int> matchingFeatureVectorsInd = searchMatchingFeatureVectors(root, featureVectors, 0.1 * featureVectors.size());

    //      for (int i = 0; i < matchingFeatureVectors.size(); i++)
    // {
    //     cout << "Contour " << i + 1 << " Fourier descriptors: ";
    //     for (int j = 0; j < matchingFeatureVectors[i].size(); j++)
    //     {
    //         cout << matchingFeatureVectors[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    vector<Rect> boundingBoxes = searchAndComputeBoundingBoxes(root, matchingFeatureVectorsInd, contours, MINBOXSIZE);

    // Loop over the detected objects and draw bounding boxes around them
    for (auto &bbox : boundingBoxes)
    {
        rectangle(image, bbox, Scalar(0, 255, 0), 2);
    }

    // Display the image with the detected objects
    imwrite("processing/7_detected objects.jpg", image);

    return 0;
}