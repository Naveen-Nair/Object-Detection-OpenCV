
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

vector<Rect> searchAndComputeBoundingBoxes(KDNode *root, vector<int> &matchingFeatureVectors, vector<vector<Point>> &contours, int MINBOXSIZE)
{

    // Search for matching feature vectors and compute bounding boxes
    vector<Rect> boundingBoxes;
    for (const auto &featureVectorInd : matchingFeatureVectors)
    {
        // Find the matching contour
        auto &matchingContour = contours[featureVectorInd];

        // Compute the bounding box of the contour
        const auto boundingBox = cv::boundingRect(matchingContour);

        // Add the bounding box to the vector if its area is greater than or equal to MINBOXSIZE
        if (boundingBox.area() >= MINBOXSIZE)
        {
            boundingBoxes.push_back(boundingBox);
        }
    }

    return boundingBoxes;
}

