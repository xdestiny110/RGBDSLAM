#include "ColorHistgram.h"

cv::MatND ColorHistCalc(const cv::Mat& image, bool changedToHSV, int bins1, int bins2, int bins3){
	float ranges1[] = { 0, 255 }, ranges2[] = { 0, 255 }, ranges3[] = { 0, 255 };
	cv::Mat img;
	if (changedToHSV){
		cv::cvtColor(image, img, cv::COLOR_BGR2HSV);
		ranges1[1] = 180 ;
	}
	else
		image.copyTo(img);

	int channels[] = { 0, 1, 2 };
	int histSize[] = { bins1, bins2, bins3 };
	const float* ranges[] = { ranges1, ranges2, ranges3 };
	
	cv::MatND hist;
	cv::calcHist(&img, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
	return hist;
}