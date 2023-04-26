#ifndef EXTRACTFEATURES_HPP
#define EXTRACTFEATURES_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;

vector<vector<double>> extractFourierDescriptors(vector<vector<Point>> contours, Mat binary_image);

#endif