
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;

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
