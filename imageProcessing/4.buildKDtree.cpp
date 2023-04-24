

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
bool compareVectors(const vector<double> &a, const vector<double> &b, int dim)
{
    return a[dim] < b[dim];
}

// Recursively build kd-tree by splitting at median of a specific dimension
KDNode *buildKdTree(vector<vector<double>> &featureVectors, int start, int end, int depth)
{
    if (start > end)
        return nullptr;
    int dim = depth % featureVectors[0].size();
    sort(featureVectors.begin() + start, featureVectors.begin() + end + 1, [dim](const vector<double> &a, const vector<double> &b)
         { return compareVectors(a, b, dim); });
    int median = start + (end - start) / 2;
    KDNode *node = new KDNode(featureVectors[median], dim);
    node->left = buildKdTree(featureVectors, start, median - 1, depth + 1);
    node->right = buildKdTree(featureVectors, median + 1, end, depth + 1);
    return node;
}

