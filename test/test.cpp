#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    Mat image;
    image = imread("../images/Marathon.jpeg", IMREAD_COLOR);
    namedWindow("Display window", WINDOW_NORMAL);
    imshow("Display window", image);
    waitKey(0);
    return 0;
}
