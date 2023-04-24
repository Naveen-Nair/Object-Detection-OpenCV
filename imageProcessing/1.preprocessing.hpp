#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;


void grayscale(Mat &image, Mat &gray_image);
void threshold(Mat &image, Mat &thresholded_image);
void gaussianBlur(const Mat &input, Mat &output, int kernel_size, double sigma);

#endif