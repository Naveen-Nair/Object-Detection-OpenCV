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

#define MINBOXSIZE 1000

int main()
{

    cv::VideoCapture cap("../videos/input0.mp4"); // open video file
    if (!cap.isOpened())
    { // check if we succeeded
        return -1;
    }

    cv::VideoWriter output("output.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 30, cv::Size(640, 480));

    cv::Mat frame;
    while (true)
    {
        cap >> frame; // get a new frame from video
        if (frame.empty())
        {
            break;
        }

        // Create a new grayscale frame with the same size as the input frame
        // CV_8UC1 - cv frame with 8 bit unsigned with only one channel (ie grayscale)
        Mat gray_frame(frame.size(), CV_8UC1);
        grayscale(frame, gray_frame);

        // create a one channel frame of the same size as frame
        Mat thresholded_frame(frame.size(), CV_8UC1);
        threshold(gray_frame, thresholded_frame);

        // Apply Gaussian blur
        Mat blurred_frame(frame.size(), CV_8UC1);
        gaussianBlur(thresholded_frame, blurred_frame, KERNEL_SIZE, SIGMA);

        Mat binary_frame(frame.size(), CV_8UC1);
        threshold(blurred_frame, binary_frame);

        // Mat grad_frame(frame.size(), CV_8UC1);
        // sobelOperator(binary_frame, grad_frame);

        vector<vector<Point>> contours;
        detectEdges(binary_frame, contours);

        vector<vector<double>> featureVectors = extractFourierDescriptors(contours, binary_frame);

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
            rectangle(frame, bbox, Scalar(0, 255, 0), 2);
        }

        output.write(frame);  // write frame to output video

        cv::imshow("Object Detection", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}