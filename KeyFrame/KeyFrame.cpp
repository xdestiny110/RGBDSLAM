#include "KeyFrame.h"
#include "Miscellaneous.h"
#include <pcl/io/ply_io.h>
#include <boost/filesystem.hpp>

string KeyFrame::savePath = "";

KeyFrame::KeyFrame(Eigen::MatrixXf pos, const cv::Mat& _rgbImage, const cv::Mat& _depthImage,
	const vector<cv::KeyPoint>& _keys, int _imgId)
	:robotPos(pos), keypoint(_keys), imgId(_imgId), rgbImage(_rgbImage), depthImage(_depthImage)
{

}

KeyFrame::KeyFrame()
{

}

void KeyFrame::write(cv::FileStorage& fs) const
{
	fs << "{" << "ID" << imgId;
	ostringstream os;
	os << imgId;
	boost::filesystem::path rgbPath = string("RGB_") + os.str() + ".png",
		depthPath = string("Depth_") + os.str() + ".png";
	fs << "RGB" << rgbPath.generic_string();
	fs << "Depth" << depthPath.generic_string();
	
	cv::imwrite(KeyFrame::savePath + rgbPath.generic_string(), rgbImage);
	cv::imwrite(KeyFrame::savePath + depthPath.generic_string(), depthImage);

	cv::Mat T(4, 4, CV_32FC1);
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++)
			T.at<float>(i, j) = robotPos(i, j);
	}
	fs << "RobotPos" << T;
	fs << "Hist" << hist;
	//fs << "HashKey" <<"[:";
	//for (auto it = hashKey.begin(); it != hashKey.end(); it++)
	//	fs << *it;
	//fs << "]";
	fs << "}";
}

void KeyFrame::read(const cv::FileNode& fn)
{
	fn["ID"] >> imgId;
	string rgbName, depthName;
	fn["RGB"] >> rgbName;
	rgbImage = cv::imread(KeyFrame::savePath + rgbName);
	fn["Depth"] >> depthName;
	depthImage = cv::imread(KeyFrame::savePath + depthName, CV_LOAD_IMAGE_ANYDEPTH);

	cv::Mat T;
	fn["RobotPos"] >> T;
	robotPos = Eigen::Matrix4f::Identity();
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++)
			robotPos(i, j) = T.at<float>(i, j);
	}

	fn["Hist"] >> hist;
	//cv::FileNode node = fn["HashKey"];
	//hashKey.clear();
	//for (auto it = node.begin(); it != node.end(); it++)
	//	hashKey.push_back(*it);
}

void KeyFrame::SaveKeyFrameList(vector<KeyFrame>& KeyFrameList,const string& path)
{
	boost::filesystem::path p = path;
	if (!boost::filesystem::is_directory(p))
		boost::filesystem::create_directory(p);
	KeyFrame::savePath = path;

	cv::FileStorage fs(path + "KeyFrame.xml", cv::FileStorage::WRITE);
	fs << "NumberOfKeyFrame" << static_cast<int>(KeyFrameList.size());
	//fs << "HashKey" << "[";
	//for (auto it = KeyFrameList.begin(); it != KeyFrameList.end(); it++){
	//	fs << it->hashKey;
	//}
	//fs << "]";
	fs << "KeyFrameList" << "[";
	for (int i = 0; i < KeyFrameList.size();i++)
		fs << KeyFrameList[i];
	
	fs << "]";
}

void KeyFrame::LoadKeyFrameList(vector<KeyFrame>& KeyFrameList, const string& path)
{
	KeyFrame::savePath = path;
	KeyFrameList.clear();
	cv::FileStorage fs(path + "KeyFrame.xml", cv::FileStorage::READ);
	int numberOfKeyFrame;
	fs["NumberOfKeyFrame"] >> numberOfKeyFrame;
	KeyFrameList.resize(numberOfKeyFrame);
	cv::FileNode fn = fs["KeyFrameList"];
	for (int i = 0; i < numberOfKeyFrame; i++){
		istringstream is(i);
		fn[is.str()] >> KeyFrameList[i];
	}
}