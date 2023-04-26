#ifndef BUILD_KDTREE_HPP
#define BUILD_KDTREE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;

struct KDNode
{
    vector<double> point;
    int splitDim;
    KDNode *left, *right;
    KDNode(vector<double> &p, int dim) : point(p), splitDim(dim), left(nullptr), right(nullptr) {}
};

// Compare function to sort feature vectors based on a specific dimension
bool compareVectors(const vector<double> &a, const vector<double> &b, int dim);

// Recursively build kd-tree by splitting at median of a specific dimension
KDNode *buildKdTree(vector<vector<double>> &featureVectors, int start, int end, int depth);


#endif