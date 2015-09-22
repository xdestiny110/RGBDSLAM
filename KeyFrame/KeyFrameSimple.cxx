#include "KeyFrameSimple.h"
#include <boost/filesystem.hpp>

void KeyFrameSimple::LoadKeyFrameList(vector<KeyFrameSimple>& keyFrameList,const string& path){
	string bathPath = path;
	cv::FileStorage fs(path + "KeyFrame.xml", cv::FileStorage::READ);
	int numberOfKeyFrame;
	fs["NumberOfKeyFrame"] >> numberOfKeyFrame;
	keyFrameList.clear();
	keyFrameList.resize(numberOfKeyFrame);
	cv::FileNode fn = fs["KeyFrameList"];
	int i = 0;
	for (auto it = fn.begin(); it != fn.end() && i < numberOfKeyFrame; it++, i++){
		*it >> keyFrameList[i];
	}
}

void KeyFrameSimple::read(const cv::FileNode& fn){
	fn["ID"] >> imgId;
	fn["RGB"] >> rgbImageName;
	fn["Depth"] >> depthImageName;
	cv::Mat pos;
	fn["RobotPos"] >> pos;
	robotPos = Eigen::Matrix4f::Identity();
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++)
			robotPos(i, j) = pos.at<float>(i, j);
	}
	fn["Hist"] >> hist;
	//cv::FileNode node = fn["HashKey"];
	//hashKey.clear();
	//for (auto it = node.begin(); it != node.end(); it++)
	//	hashKey.push_back(*it);
}

void KeyFrameSimple::write(cv::FileStorage& fs) const
{
	fs << "{" << "ID" << imgId;
	ostringstream os;
	os << imgId;
	boost::filesystem::path rgbPath = string("RGB_") + os.str() + ".png",
		depthPath = string("Depth_") + os.str() + ".png";
	fs << "RGB" << rgbPath.generic_string();
	fs << "Depth" << depthPath.generic_string();

	cv::Mat T(4, 4, CV_32FC1);
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++)
			T.at<float>(i, j) = robotPos(i, j);
	}
	fs << "RobotPos" << T;
	fs << "Hist" << hist;
	//fs << "HashKey" << "[:";
	//for (auto it = hashKey.begin(); it != hashKey.end(); it++)
	//	fs << *it;
	//fs << "]";
	fs << "}";
}

void KeyFrameSimple::SaveKeyFrameList(const vector<KeyFrameSimple>& KeyFrameList, const string& path){
	cv::FileStorage fs(path + "KeyFrame.xml", cv::FileStorage::WRITE);
	fs << "NumberOfKeyFrame" << static_cast<int>(KeyFrameList.size());
	//fs << "HashKey" << "[";
	//for (auto it = KeyFrameList.begin(); it != KeyFrameList.end(); it++){
	//	fs << it->hashKey;
	//}
	//fs << "]";
	fs << "KeyFrameList" << "[";
	for (int i = 0; i < KeyFrameList.size(); i++)
		fs << KeyFrameList[i];

	fs << "]";
}
