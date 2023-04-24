#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>
#include <stack>

#include <iostream>

using namespace cv;
using namespace std;

#define KERNEL_SIZE 3
#define SIGMA 1
#define IMG_PATH "../images/Football.jpeg"
#define OPERATOR >
#define MINBOXSIZE 100000

void grayscale(Mat &image, Mat &gray_image)
{
    // Loop through each pixel in the input image and convert to grayscale
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            // vec3b is the channel to access the r g b values of the pixel
            Vec3b pixel = image.at<Vec3b>(i, j);

            // multiplying with the luma coefficients to convert each pixel into grayscale
            uchar gray_value = 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0];

            // uchar is the function to access the grayscale value of the pixel
            gray_image.at<uchar>(i, j) = gray_value;
        }
    }
}

void threshold(Mat &image, Mat &thresholded_image)
{

    // Calculate histogram
    int hist[256] = {0};
    int num_pixels = image.rows * image.cols;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int intensity = (int)image.at<uchar>(i, j);
            hist[intensity]++;
        }
    }

    // Normalize histogram
    double norm_hist[256] = {0};
    for (int i = 0; i < 256; i++)
    {
        norm_hist[i] = (double)hist[i] / num_pixels;
    }

    // Calculate cumulative sums
    double cum_sum[256] = {0};
    cum_sum[0] = norm_hist[0];
    for (int i = 1; i < 256; i++)
    {
        cum_sum[i] = cum_sum[i - 1] + norm_hist[i];
    }

    // Calculate inter-class variance for each threshold
    double max_variance = 0;
    int threshold = 0;
    for (int i = 0; i < 256; i++)
    {
        double w0 = cum_sum[i];
        double w1 = 1 - w0;
        double mean0 = 0;
        double mean1 = 0;
        for (int j = 0; j <= i; j++)
        {
            mean0 += j * norm_hist[j] / w0;
        }
        for (int j = i + 1; j < 256; j++)
        {
            mean1 += j * norm_hist[j] / w1;
        }
        double variance = w0 * w1 * pow((mean0 - mean1), 2);
        if (variance > max_variance)
        {
            max_variance = variance;
            threshold = i;
        }
    }
    cout << "Threshold" << threshold << endl;
    // Threshold the image
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) OPERATOR threshold)
            {
                thresholded_image.at<uchar>(i, j) = 255; // white
            }
            else
            {
                thresholded_image.at<uchar>(i, j) = 0; // black
            }
        }
    }
}

void gaussianBlur(const Mat &input, Mat &output, int kernel_size, double sigma)
{
    int k = (kernel_size - 1) / 2;
    double kernel[kernel_size][kernel_size];
    double sum = 0.0;

    // Create Gaussian kernel
    for (int x = -k; x <= k; x++)
    {
        for (int y = -k; y <= k; y++)
        {
            double value = exp(-(x * x + y * y) / (2.0 * sigma * sigma));
            kernel[x + k][y + k] = value;
            sum += value;
        }
    }

    // Normalize kernel
    for (int x = 0; x < kernel_size; x++)
    {
        for (int y = 0; y < kernel_size; y++)
        {
            kernel[x][y] /= sum;
        }
    }

    // Apply convolution
    output.create(input.size(), input.type());
    for (int i = k; i < input.rows - k; i++)
    {
        for (int j = k; j < input.cols - k; j++)
        {
            double sum = 0.0;
            for (int x = -k; x <= k; x++)
            {
                for (int y = -k; y <= k; y++)
                {
                    sum += kernel[x + k][y + k] * input.at<uchar>(i + x, j + y);
                }
            }
            output.at<uchar>(i, j) = sum;
        }
    }
}

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
    for (size_t i = 0; i < contours.size(); i++)
    {
        // Generate a random color for each contour
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // Approximate the contour with a polygonal curve
        vector<Point> poly;
        approxPolyDP(contours[i], poly, 3, true);

        // Draw the polygonal curve
        drawContours(drawing, vector<vector<Point>>{poly}, 0, color, 2);
    }

    // Display the image with the contours
    imwrite("processing/6_contours.jpg", drawing);
}

vector<vector<double>> extractFourierDescriptors(vector<vector<Point>> contours, Mat binary_image)
{
    vector<vector<double>> featureVectors;
    for (int i = 0; i < contours.size(); i++)
    {
        // Compute Fourier descriptors
        vector<Point> contour = contours[i];
        Mat contourMat;
        contourMat.create(Size(contour.size(), 1), CV_32FC2);
        for (int j = 0; j < contour.size(); j++)
        {
            contourMat.at<Vec2f>(j) = Vec2f(contour[j].x, contour[j].y);
        }
        Mat fourierCoeffs;
        dft(contourMat, fourierCoeffs, DFT_COMPLEX_OUTPUT);
        int numRows = fourierCoeffs.rows;
        int maxRow = (contour.size() + 1) / 2;
        if (maxRow > numRows)
        {
            maxRow = numRows;
        }
        fourierCoeffs = fourierCoeffs.rowRange(0, maxRow);
        if (fourierCoeffs.cols > 1)
        {
            fourierCoeffs.col(1) *= -1;
        }
        vector<double> fd;
        for (int j = 0; j < fourierCoeffs.rows; j++)
        {
            fd.push_back(norm(fourierCoeffs.at<Vec2f>(j)));
        }
        featureVectors.push_back(fd);
    }

    return featureVectors;
}

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

vector<Rect> searchAndComputeBoundingBoxes(KDNode *root, vector<int> &matchingFeatureVectors, vector<vector<Point>> &contours)
{
    // vector<Rect> boundingBoxes;

    // Search for matching feature vectors and compute bounding boxes
    std::vector<cv::Rect> boundingBoxes;
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

int main()
{
    // Load the image
    Mat image = imread(IMG_PATH);

    // Create a new grayscale image with the same size as the input image
    // CV_8UC1 - cv image with 8 bit unsigned with only one channel (ie grayscale)
    Mat gray_image(image.size(), CV_8UC1);

    grayscale(image, gray_image);

    // Save the gray image
    imwrite("processing/1_gray_image_function.jpg", gray_image);

    // create a one channel image of the same size as image
    Mat thresholded_image(image.size(), CV_8UC1);

    // 127 being in the middle of 0 to 255, so generally we threshold using that
    threshold(gray_image, thresholded_image);
    imwrite("processing/2_thresholded_image.jpg", thresholded_image);

    // Apply Gaussian blur
    Mat blurred_image(image.size(), CV_8UC1);
    gaussianBlur(thresholded_image, blurred_image, KERNEL_SIZE, SIGMA);
    // GaussianBlur(thresholded_image, blurred_image, Size(3,3), 0);
    imwrite("processing/3_processed_image2.jpg", blurred_image);

    Mat binary_image(image.size(), CV_8UC1);

    // 127 being in the middle of 0 to 255, so generally we threshold using that
    threshold(blurred_image, binary_image);
    imwrite("processing/4_thresholded_image2.jpg", binary_image);

    Mat grad_image(image.size(), CV_8UC1);
    sobelOperator(binary_image, grad_image);

    vector<vector<Point>> contours;
    detectEdges(grad_image, contours);

    vector<vector<double>> featureVectors = extractFourierDescriptors(contours, binary_image);

    // for (int i = 0; i < featureVectors.size(); i++)
    // {
    //     cout << "Contour " << i + 1 << " Fourier descriptors: ";
    //     for (int j = 0; j < featureVectors[i].size(); j++)
    //     {
    //         cout << featureVectors[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    KDNode *root = buildKdTree(featureVectors, 0, featureVectors.size() - 1, 0);

    vector<int> matchingFeatureVectorsInd = searchMatchingFeatureVectors(root, featureVectors, 0.1 * featureVectors.size());

    //      for (int i = 0; i < matchingFeatureVectors.size(); i++)
    // {
    //     cout << "Contour " << i + 1 << " Fourier descriptors: ";
    //     for (int j = 0; j < matchingFeatureVectors[i].size(); j++)
    //     {
    //         cout << matchingFeatureVectors[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    vector<Rect> boundingBoxes = searchAndComputeBoundingBoxes(root, matchingFeatureVectorsInd, contours);

    // Loop over the detected objects and draw bounding boxes around them
    for (auto &bbox : boundingBoxes)
    {
        rectangle(image, bbox, Scalar(0, 255, 0), 2);
    }

    // Display the image with the detected objects
    imwrite("processing/7_detected objects.jpg", image);

    return 0;
}