#ifndef BOUNDINGBOXES_HPP
#define BOUNDINGBOXES_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;



vector<Rect> searchAndComputeBoundingBoxes(KDNode *root, vector<int> &matchingFeatureVectors, vector<vector<Point>> &contours, int MINBOXSIZE);

#endif