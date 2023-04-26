
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;

#define OPERATOR <

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

    int threshold = 90;

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
