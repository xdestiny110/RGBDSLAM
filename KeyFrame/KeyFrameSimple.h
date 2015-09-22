#ifndef KEYFRAMESIMPLE_H_
#define KEYFRAMESIMPLE_H_

#include "common.h"

class KeyFrameSimple{
public:
	KeyFrameSimple(){}
	static void LoadKeyFrameList(vector<KeyFrameSimple>&,const string&);
	void read(const cv::FileNode&);
	static void SaveKeyFrameList(const vector<KeyFrameSimple>&, const string&);
	void write(cv::FileStorage& fs) const;
	string rgbImageName, depthImageName;
	//vector<int> hashKey;
	cv::MatND hist;
	Eigen::MatrixXf robotPos;
	int imgId;
};

static void read(const cv::FileNode& fn, KeyFrameSimple& kf, const KeyFrameSimple& default_value = KeyFrameSimple())
{
	if (fn.empty())
		kf = default_value;
	else
		kf.read(fn);
}

static void write(cv::FileStorage& fs, const string&, const  KeyFrameSimple& kf)
{
	kf.write(fs);
}

#endif