#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // Load the image
    Mat image = imread("../images/Marathon.jpeg");

    // Convert the image to grayscale
    
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

       // Save the processed image
    imwrite("processing/gray_image.jpg", gray_image);

    // Apply thresholding
    Mat thresholded_image;
    threshold(gray_image, thresholded_image, 128, 255, THRESH_BINARY);
     imwrite("processing/thresholded_image.jpg", thresholded_image);

    // Apply Gaussian blur
    Mat blurred_image;
    GaussianBlur(thresholded_image, blurred_image, Size(3,3), 0);

 
   
    imwrite("processing/processed_image.jpg", blurred_image);

    return 0;
}