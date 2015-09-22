#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include "common.h"

class KeyFrame
{
public:
	KeyFrame();
	KeyFrame(Eigen::MatrixXf, const cv::Mat&, const cv::Mat&,
		const vector<cv::KeyPoint>&, int);

	bool operator<(const KeyFrame& other) const
	{
		return imgId < other.imgId;
	}

	void write(cv::FileStorage&) const;
	void read(const cv::FileNode&);
	static void SaveKeyFrameList(vector<KeyFrame>&, const string&);
	static void LoadKeyFrameList(vector<KeyFrame>&, const string&);
	Eigen::MatrixXf robotPos;
	int imgId;
	//vector<int> hashKey;
	cv::MatND hist;
	cv::Mat rgbImage, depthImage;
	vector<cv::KeyPoint> keypoint;
	static string savePath;
};

static void write(cv::FileStorage& fs, const string&, const KeyFrame& kf)
{
	kf.write(fs);
}

static void read(const cv::FileNode& fn, KeyFrame& kf, const KeyFrame& default_value = KeyFrame())
{
	if (fn.empty())
		kf = default_value;
	else
		kf.read(fn);

}
#endif