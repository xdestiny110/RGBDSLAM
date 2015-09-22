#ifndef LOCATION_H_
#define LOCATION_H_

#include <vector>
#include <set>
#include <string>
#include <opencv2/opencv.hpp>
#include "ColorHistgram.h"
#include "RegistrationBase.h"
#include "KeyFrameSimple.h"

class Location
{
public:
	explicit Location(const string& path) :bathPath(path){
		KeyFrameSimple::LoadKeyFrameList(keyFrameList, path);
	}
	Location& SetObject(const cv::Mat&, const cv::Mat&);
	Eigen::MatrixXf FindKeyFrame(int = 0);
private:
	RegistrationBase::Ptr reg;
	vector<KeyFrameSimple> keyFrameList;
	set<pair<float, int>> sortedHashKeyList;
	string bathPath;
	cv::Mat objectRGBImage, objectDepthImage;
	//vector<int> objectHashKey;
	cv::MatND objectHist;
};

#endif