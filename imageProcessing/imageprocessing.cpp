#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define THRESHOLD_VAlUE 127
#define KERNEL_SIZE 3
#define SIGMA 1

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

void gaussianBlur(const cv::Mat& input, cv::Mat& output, int kernel_size, double sigma)
{
    int k = (kernel_size - 1) / 2;
    double kernel[kernel_size][kernel_size];
    double sum = 0.0;

    // Create Gaussian kernel
    for (int x = -k; x <= k; x++) {
        for (int y = -k; y <= k; y++) {
            double value = exp(-(x*x + y*y) / (2.0*sigma*sigma));
            kernel[x + k][y + k] = value;
            sum += value;
        }
    }

    // Normalize kernel
    for (int x = 0; x < kernel_size; x++) {
        for (int y = 0; y < kernel_size; y++) {
            kernel[x][y] /= sum;
        }
    }

    // Apply convolution
    output.create(input.size(), input.type());
    for (int i = k; i < input.rows - k; i++) {
        for (int j = k; j < input.cols - k; j++) {
            double sum = 0.0;
            for (int x = -k; x <= k; x++) {
                for (int y = -k; y <= k; y++) {
                    sum += kernel[x + k][y + k] * input.at<uchar>(i + x, j + y);
                }
            }
            output.at<uchar>(i, j) = sum;
        }
    }
}

void detectEdges(Mat& image, vector<vector<Point>>& contours)
{
    // Find contours
    vector<Vec4i> hierarchy;
    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Draw contours
    Mat drawing = Mat::zeros(image.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(0, 255, 0); // Green color
        drawContours(drawing, contours, i, color, 2, LINE_8, hierarchy, 0);
    }
    imwrite("processing/contours.jpg", drawing);
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
    threshold(gray_image, thresholded_image, THRESHOLD_VAlUE);
    imwrite("processing/thresholded_image2.jpg", thresholded_image);

    // Apply Gaussian blur
    Mat blurred_image(image.size(), CV_8UC1);
    gaussianBlur(thresholded_image, blurred_image, KERNEL_SIZE, SIGMA);
    // GaussianBlur(thresholded_image, blurred_image, Size(3,3), 0);
    imwrite("processing/processed_image2.jpg", blurred_image);

    Mat binary_image(image.size(), CV_8UC1);
    
    //127 being in the middle of 0 to 255, so generally we threshold using that
    threshold(blurred_image, binary_image, THRESHOLD_VAlUE);
    imwrite("processing/thresholded_image2.jpg", thresholded_image);

    vector<vector<Point>> contours;
    detectEdges(binary_image, contours);
 
   
   

    return 0;
}