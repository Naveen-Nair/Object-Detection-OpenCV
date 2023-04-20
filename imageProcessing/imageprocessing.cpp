#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void grayscale(Mat &image, Mat &gray_image) {
   // Loop through each pixel in the input image and convert to grayscale
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {

            //vec3b is the channel to access the r g b values of the pixel
            Vec3b pixel = image.at<Vec3b>(i, j);

            //multiplying with the luma coefficients to convert each pixel into grayscale
            uchar gray_value = 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0];

            //uchar is the function to access the grayscale value of the pixel
            gray_image.at<uchar>(i, j) = gray_value;
        }
    }
}

void threshold(Mat &image, Mat &thresholded_image, int threshold_value){

    
     for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) > threshold_value) {
                thresholded_image.at<uchar>(i, j) = 255;
            }
            else {
                thresholded_image.at<uchar>(i, j) = 0;
            }
        }
    }
}

int main()
{
    // Load the image
    Mat image = imread("../images/Marathon.jpeg");


    // Create a new grayscale image with the same size as the input image
    //CV_8UC1 - cv image with 8 bit unsigned with only one channel (ie grayscale)
    Mat gray_image(image.size(), CV_8UC1);

    grayscale(image, gray_image);
 
    // Save the gray image
    imwrite("processing/gray_image_function.jpg", gray_image);


    // create a one channel image of the same size as image
    Mat thresholded_image(image.size(), CV_8UC1);
    
    //127 being in the middle of 0 to 255, so generally we threshold using that
    threshold(gray_image, thresholded_image, 127);
    imwrite("processing/thresholded_image2.jpg", thresholded_image);

    // Apply Gaussian blur
    Mat blurred_image;
    GaussianBlur(thresholded_image, blurred_image, Size(3,3), 0);

 
   
    imwrite("processing/processed_image2.jpg", blurred_image);

    return 0;
}