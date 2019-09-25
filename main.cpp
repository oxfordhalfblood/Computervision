
/*  Student Name  : Afrida Tabassum
Student Number: 6015906
E-mail Address : at057@uowmail.edu.au

Assignment 1
CSCI935 - Computer Vision
Created by Afrida on 24/8/19.
Copyright © 2019 Afrida. All rights reserved.

 ######### Task1:  #########
 
 This reads an image and display the original image in color and its components in different color spaces, as CIE-XYZ, CIE-Lab, YCrCb and HSB. Below functions are used in this task:
 
    taskOne(string conversionCode, Mat originalJPG ):
 
 Takes the code from user and I used color cvtColor() for colorconversion. Then splitted the images into different channels(BGR), normalized them from 0 to 1, then the desired component(either X or Y or Z) will be merged into a vector and the converted images will be displayed in a single window.
 
 ######### Task2:  #########
 
 This implements the color image processing chain that converts raw CMOS image sensor created image raw image data into true color RGB images. At 1st we interpolate the image using Bayer pattern. For that I used a forloop which considers the index from 1 to end-2, so my algorithm is efficient as it doesn't go out of boundary for the corner indices as matrix indices are from 0 to (end-1).
 
 
 Then the bilinear interpolated image is passed onto the colorcorrection() function and a Matrix multiplication is done within the given matrix and that image.
 
 Then colorcorrected image is passed onto gammacorrection() and a lookup table is formulated using ((input/255)^G)*255+0.5, which maps the input pixel values to the output gamma corrected values and //scaling values in the range 0 - 255//scaling values in the range 0 - 255scaling values in the range 0 - 255 I then used the table to quickly determine the output value for a given pixel in O(1) time.
 
*/

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/mat.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void taskOne(string, Mat);
void taskTwo(Mat);
Mat bilinearInterpolation(Mat);
Mat colourCorrection(Mat);
Mat gammaCorrection(Mat);
void displayImages(Mat, Mat, Mat, Mat);

//merging different channels
vector<Mat> mergeChannels(Mat);

int main(int argc, char** argv) {
    
    Mat originalJPG, originalBMP;
    
    if (argc > 3 || argc < 2){
        cout << " Execution Syntax Error - use (-.exe bmpfile) format" << endl;
        cout << "                              (-.exe  -XYZ|-Lab|-YCrCb|-HSB imagefile) format" << endl;
        return -1;
    }
    else if (argc == 3) {
        originalJPG = imread(argv[2], IMREAD_COLOR);
        
        if (!originalJPG.data)
        {
            cout << "Could not open or find the image" << endl;
            return -1;
        }
        //task one function
        taskOne(argv[1], originalJPG);
    }
    else if (argc == 2) {
        originalBMP = imread(argv[1], IMREAD_COLOR);
        if (!originalBMP.data)
        {
            cout << "Could not open or find the image" << endl;
            return -1;
        }
        //task two function
        taskTwo(originalBMP);
    }
    
    return 0;
}

vector<Mat> mergeChannels(Mat currentChannel) {
    vector<Mat> vector;
    vector.push_back(currentChannel);
    vector.push_back(currentChannel);
    vector.push_back(currentChannel);
    
    return vector;
}

void taskOne(string conversionCode, Mat originalJPG ) {
    cout<<"Conversion Code "<<string(conversionCode)<<endl;
    vector<Mat> matVector;
    Mat component1, component2, component3, image, convImage, channel[3];
    
    originalJPG.convertTo(image, CV_32F);
    image *= 1.0 / 255;  // scaling 0 to 255
    
    //checking the arguments passed as conversion code
    bool isXYZ = (string(conversionCode) == "-XYZ");
    bool isLab = string(conversionCode) == "-Lab";
    bool isYCrCb = string(conversionCode) == "-YCrCb";
    bool isHSB = string(conversionCode) == "-HSB";
    
    if (isXYZ) {
        cvtColor(image, convImage, COLOR_BGR2XYZ);
    }
    else if (isLab) {
        cvtColor(image, convImage, COLOR_BGR2Lab);
    }
    else if (isYCrCb) {
        cvtColor(image, convImage, COLOR_BGR2YCrCb);
    }
    else if (isHSB) {
        cvtColor(image, convImage, COLOR_BGR2HSV);
    }
    else {
        cout << "Error! Use one of the below color spaces\n";
        cout << " -XYZ | -Lab | -YCrCb | -HSB" << endl;
        exit (0);
    }
    
    //spliting images into different channels
    split(convImage, channel);
    
  
    
    // normalizing from 0 to 1
    float max = 1.0, min = 0.0;
    float intensity = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int row = 0; row < channel[i].rows; row++) {
            for (int col = 0; col < channel[i].cols; col++) {
                intensity = channel[i].at<float>(row, col);
                //find minimum
                if (intensity < min) {
                    min = intensity;
                }
            }
        }
        
        //subtract from pixels, if min is less than 0. so we make that pixel zero
        if (min < 0) {
            cout<<min;
            for (int row = 0; row < channel[i].rows; row++) {
                for (int col = 0; col < channel[i].cols; col++) {
                    channel[i].at<float>(row, col) = channel[i].at<float>(row, col) - min;
                }
            }
        }
        
        float in = 0.0, max = 0.0;
        
        //find max
        for (int row = 0; row < channel[i].rows; row++) {
            for (int col = 0; col < channel[i].cols; col++) {
                in = channel[i].at<float>(row, col);
                if (in > max) {
                    max = in;
                }
            }
        }
        
        //divide with max
        if (max > 1) {
            for (int row = 0; row < channel[i].rows; row++) {
                for (int col = 0; col < channel[i].cols; col++) {
                    channel[i].at<float>(row, col) = channel[i].at<float>(row, col) / max;
                }
            }
        }
    }
    
    
    //components and merging
    for (int i = 0; i < 3; i++) {
        matVector = mergeChannels(channel[i]);
        
        if (i == 0) {
            merge(matVector, component1);
        }
        else if (i == 1) {
            merge(matVector, component2);
        }
        else {
            merge(matVector, component3);
        }
        
        matVector.clear();
    }
    
    if(isHSB)
        displayImages(image, component3, component1, component2);
    else
    //display the images in a single window
        displayImages(image, component1, component2, component3);
}

void taskTwo(Mat originalBMP){
    
    Mat imageBI = bilinearInterpolation(originalBMP);
    Mat imageCC = colourCorrection(imageBI);
    Mat imageGC = gammaCorrection(imageCC);
    
    displayImages(originalBMP, imageBI, imageCC, imageGC);
}

/*
 BGR; B=vec[0], g=vec[1], r=vec[2].
 */
Mat bilinearInterpolation(Mat imgOriginalBMP) {
    
    Mat img;
    imgOriginalBMP.copyTo(img);
    
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            if (i % 2 != 0 && j % 2 != 0) {
                //interpolate the green
                //blue
                img.at<Vec3b>(i, j)[0] = (img.at<Vec3b>(i, j - 1)[0] + img.at<Vec3b>(i, j + 1)[0]) / 2;
                //red
                img.at<Vec3b>(i, j)[2] = (img.at<Vec3b>(i - 1, j)[2] + img.at<Vec3b>(i + 1, j)[2]) / 2;
            }
            else if (i % 2 == 0 && j % 2 != 0) {
                 //interpolate the red
                //4 blue
                img.at<Vec3b>(i, j)[0] = (img.at<Vec3b>(i - 1, j - 1)[0] + img.at<Vec3b>(i - 1, j + 1)[0] + img.at<Vec3b>(i + 1, j - 1)[0] + img.at<Vec3b>(i + 1, j + 1)[0]) / 4;
                //4 green
                img.at<Vec3b>(i, j)[1] = (img.at<Vec3b>(i - 1, j)[1] + img.at<Vec3b>(i + 1, j)[1] + img.at<Vec3b>(i, j - 1)[1] + img.at<Vec3b>(i, j + 1)[1]) / 4;
            }
            //interpolate the blue
           
            else if (i % 2 != 0 && j % 2 == 0) {
                 //4 green
                img.at<Vec3b>(i, j)[1] = (img.at<Vec3b>(i - 1, j)[0] + img.at<Vec3b>(i + 1, j)[0] + img.at<Vec3b>(i, j - 1)[0] + img.at<Vec3b>(i, j + 1)[0]) / 4;
                 //4 red
                img.at<Vec3b>(i, j)[2] = (img.at<Vec3b>(i - 1, j - 1)[2] + img.at<Vec3b>(i - 1, j + 1)[2] + img.at<Vec3b>(i + 1, j - 1)[2] + img.at<Vec3b>(i + 1, j + 1)[2]) / 4;
            }
            
            //diagonals
            
            else {
                img.at<Vec3b>(i, j)[0] = (img.at<Vec3b>(i - 1, j)[0] + img.at<Vec3b>(i + 1, j)[0]) / 2;
                img.at<Vec3b>(i, j)[2] = (img.at<Vec3b>(i, j - 1)[2] + img.at<Vec3b>(i, j + 1)[2]) / 2;
            }
        }
    }
    return img;
}

Mat colourCorrection(Mat img) {
    
    Mat imgColorCorrected(img.size(), CV_8UC3);
    
    //invoking matrix operation from the given values
    
    Mat correctionKernel = (Mat_<float>(3, 3) << 1.18, -0.05, -0.13, -0.24, 1.29, -0.05, -0.18, -0.44, 1.62);
    Mat tempImgCFAInterpol;
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            //CV_32FC3 is a three channel matrix of 32-bit floats.converting img to tmparray data type
            Mat(img.at<Vec3b>(i, j)).convertTo(tempImgCFAInterpol, CV_32FC3);
            imgColorCorrected.at<Vec3b>(i, j) = Mat(correctionKernel * tempImgCFAInterpol);
            
        }
    }
    return imgColorCorrected;
}

Mat gammaCorrection(Mat img) {
    unsigned char lookUpTable[256];
    //gamma value = 0.5
    //scaling values in the range 0 - 255
    
 
    for (int i = 0; i < 256; ++i) {
        lookUpTable[i] = pow(i / 255.0, 0.5) * 255.0 + 0.5f;
    }
    int row, col;
    Mat GammaCorrection(img.size(), CV_8UC3);
    for (row = 0; row < img.rows; row++) {
        for (col = 0; col < img.cols; col++) {
            //blue
            //lookup tale index isn't floating point.
            GammaCorrection.at<Vec3b>(row, col)[0] = lookUpTable[img.at<Vec3b>(row, col)[0]];
            
            //green
            GammaCorrection.at<Vec3b>(row, col)[1] = lookUpTable[img.at<Vec3b>(row, col)[1]];
            //red
            GammaCorrection.at<Vec3b>(row, col)[2] = lookUpTable[img.at<Vec3b>(row, col)[2]];
        }
    }
    return GammaCorrection;
}

void displayImages(Mat image1, Mat image2, Mat image3, Mat image4) {
    Mat imghor1,imghor2,ver;
    //combining images to display them in a single window
    Size size = image1.size();
    Mat combinedImage(size.height + size.height, size.width + size.width, image1.type());
    // hconcat(image1,image2,imghor1);
    // hconcat(image3,image4,imghor2);
    // vconcat(imghor1,imghor2,ver);
    
   image1.copyTo(combinedImage(Rect(0, 0, size.width, size.height)));
   image2.copyTo(combinedImage(Rect(size.width, 0, size.width, size.height)));
   image3.copyTo(combinedImage(Rect(0, size.height, size.width, size.height)));
   image4.copyTo(combinedImage(Rect(size.width, size.height, size.width, size.height)));
    
    namedWindow("Final", WINDOW_AUTOSIZE);
    
    imshow("Final", ver);
    
    waitKey(0);
}



