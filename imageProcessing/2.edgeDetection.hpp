#ifndef EDGEDETECTION_HPP
#define EDGEDETECTION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;


void sobelOperator(const Mat &image, Mat &grad);

void detectEdges(Mat &grad, vector<vector<Point>> &contours);

#endif