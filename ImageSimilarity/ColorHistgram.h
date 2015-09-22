#ifndef COLORHISTOGRAMS_H_
#define COLORHISTOGRAMS_H_

#include <opencv2/opencv.hpp>
#include <string>
using namespace std;

cv::MatND ColorHistCalc(const cv::Mat&, bool = false, int = 32, int = 32, int = 32);



#endif