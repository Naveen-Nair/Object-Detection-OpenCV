#ifndef MATHCINGFEATURES_CPP
#define MATHCINGFEATURES_CPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <stack>
using namespace cv;
using namespace std;



vector<double> searchKDTree(KDNode *root, vector<double> &queryPoint);

vector<int> searchMatchingFeatureVectors(KDNode *root, vector<vector<double>> &featureVectors, double threshold);

double distance(const vector<double> &v1, const vector<double> &v2);

#endif