

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;


void sobelOperator(const Mat &image, Mat &grad)
{
    // Create kernels for Sobel operator
    const int kernelSize = 3;
    int kernelX[kernelSize][kernelSize] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernelY[kernelSize][kernelSize] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Compute gradients
    Mat gradX(image.size(), CV_32FC1);
    Mat gradY(image.size(), CV_32FC1);
    for (int i = 1; i < image.rows - 1; i++)
    {
        for (int j = 1; j < image.cols - 1; j++)
        {
            float gx = 0, gy = 0;
            for (int k = 0; k < kernelSize; k++)
            {
                for (int l = 0; l < kernelSize; l++)
                {
                    gx += kernelX[k][l] * image.at<uchar>(i + k - 1, j + l - 1);
                    gy += kernelY[k][l] * image.at<uchar>(i + k - 1, j + l - 1);
                }
            }
            gradX.at<float>(i, j) = gx;
            gradY.at<float>(i, j) = gy;
        }
    }

    // Compute gradient magnitude
    Mat mag(gradX.size(), CV_32FC1);
    for (int i = 0; i < mag.rows; i++)
    {
        for (int j = 0; j < mag.cols; j++)
        {
            float gx = gradX.at<float>(i, j);
            float gy = gradY.at<float>(i, j);
            mag.at<float>(i, j) = sqrt(gx * gx + gy * gy);
        }
    }

    // Normalize gradient magnitude to range [0, 255]
    normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1);

    // Set output to gradient magnitude
    grad = mag.clone();

    imwrite("processing/5_grad_x.jpg", gradX);
    imwrite("processing/5_grad_y.jpg", gradY);
    imwrite("processing/5_grad.jpg", grad);
}

void detectEdges(Mat &grad, vector<vector<Point>> &contours)
{

    // Find contours in the gradient image
    findContours(grad, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Draw the contours on the gradient image
    Mat drawing = Mat::zeros(grad.size(), CV_8UC3);
    RNG rng(12345); // Random number generator
    int thickness = 2;
    for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        for (size_t j = 0; j < contours[i].size(); j++) {
            Point p1 = contours[i][j];
            Point p2;
            if (j < contours[i].size() - 1) {
                p2 = contours[i][j + 1];
            } else {
                p2 = contours[i][0];
            }
            line(drawing, p1, p2, color, thickness);
        }
    }


    // Display the image with the contours
    imwrite("processing/6_contours.jpg", drawing);
}

