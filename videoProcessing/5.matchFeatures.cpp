
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <stack>
using namespace cv;
using namespace std;

struct KDNode
{
    vector<double> point;
    int splitDim;
    KDNode *left, *right;
    KDNode(vector<double> &p, int dim) : point(p), splitDim(dim), left(nullptr), right(nullptr) {}
};

double distance(const vector<double> &v1, const vector<double> &v2)
{
    double dist = 0.0;
    for (size_t i = 0; i < v1.size(); i++)
    {
        double diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}



vector<double> searchKDTree(KDNode *root, vector<double> &queryPoint)
{
    double bestDist = numeric_limits<double>::max();
    KDNode *bestNode = nullptr;
    vector<double> bestPoint;

    stack<KDNode *> nodesToVisit;
    nodesToVisit.push(root);

    while (!nodesToVisit.empty())
    {
        KDNode *currNode = nodesToVisit.top();
        nodesToVisit.pop();

        double currDist = distance(queryPoint, currNode->point);
        if (currDist < bestDist)
        {
            bestDist = currDist;
            bestNode = currNode;
            bestPoint = currNode->point;
        }

        double splitVal = currNode->point[currNode->splitDim];
        double queryVal = queryPoint[currNode->splitDim];

        double currBestDist = bestDist;

        if (queryVal < splitVal && currNode->left != nullptr)
        {
            nodesToVisit.push(currNode->left);
        }
        else if (queryVal >= splitVal && currNode->right != nullptr)
        {
            nodesToVisit.push(currNode->right);
        }

        double otherChildDist = fabs(queryVal - splitVal);
        if (otherChildDist < currBestDist && ((queryVal < splitVal && currNode->right != nullptr) || (queryVal >= splitVal && currNode->left != nullptr)))
        {
            if (queryVal < splitVal)
            {
                nodesToVisit.push(currNode->right);
            }
            else
            {
                nodesToVisit.push(currNode->left);
            }
        }
    }

    return bestPoint;
}

vector<int> searchMatchingFeatureVectors(KDNode *root, vector<vector<double>> &featureVectors, double threshold)
{
    vector<int> matchingFeatureVectors;

    for (int i=0; i<featureVectors.size(); i++)
    {
        auto featureVector = featureVectors[i];
        auto bestPoint = searchKDTree(root, featureVector);
        auto dist = distance(featureVector, bestPoint);
        if (dist <= threshold)
        {
            matchingFeatureVectors.push_back(i);
        }
    }

    return matchingFeatureVectors;
}

