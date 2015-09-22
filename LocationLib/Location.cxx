#include "Location.h"
#include "ColorHistgram.h"
#include "SiftGPURegistration.h"
#include "Miscellaneous.h"
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp6d.h>

Location& Location::SetObject(const cv::Mat& _rgbImage, const cv::Mat& _depthImage){
	objectRGBImage = _rgbImage;
	objectDepthImage = _depthImage;
	cv::MatND objectHashKey = ColorHistCalc(_rgbImage);
	sortedHashKeyList.clear();
	for (auto it = keyFrameList.begin(); it != keyFrameList.end(); it++){
		//sortedHashKeyList.insert(pair<double, int>(ColorHistCalc(objectHashKey, it->hashKey),
		//it - keyFrameList.begin()));
		sortedHashKeyList.insert(pair<double, int>(1 - cv::compareHist(objectHashKey, it->hist, CV_COMP_CORREL),
			it - keyFrameList.begin()));
	}
		
	return *this;
}

Eigen::MatrixXf Location::FindKeyFrame(int ind){
	set<pair<int, int>> s;
	int times = 0;

	for (auto it = sortedHashKeyList.begin(); it != sortedHashKeyList.end() && times < 20; it++, times++){
		cv::Mat sceneRGBImage = cv::imread(bathPath + keyFrameList[it->second].rgbImageName),
			sceneDepthImage = cv::imread(bathPath + keyFrameList[it->second].depthImageName, cv::IMREAD_ANYDEPTH);
		if (reg.use_count() == 0)
			reg = make_shared<SiftGPURegistration>(objectRGBImage, sceneRGBImage,
			objectDepthImage, sceneDepthImage);
		else{
			if (it == sortedHashKeyList.begin())
				reg->SetObject(objectRGBImage, objectDepthImage);
			reg->SetScene(sceneRGBImage, sceneDepthImage);
		}

		reg->Apply();
		reg->ApplyWith5Pt();
		cout << ind << " to #" << it->second << " 5 points method: " << reg->GetInliersNum()
			<< " Similirity: " << it->first << endl;
		if (reg->Get5PointInlierNum() > 15 && reg->Get5PointInlierWithDepthNum() > 5){
			return keyFrameList[it->second].robotPos*reg->GetTransformationMatrix();
		}
	}

	//for (auto it = sortedHashKeyList.begin(); it != sortedHashKeyList.end() /*&& times < 20*/; it++, times++){
	//	cv::Mat sceneRGBImage = cv::imread(bathPath + keyFrameList[it->second].rgbImageName),
	//		sceneDepthImage = cv::imread(bathPath + keyFrameList[it->second].depthImageName, cv::IMREAD_ANYDEPTH);
	//	if (reg.use_count() == 0)
	//		reg = make_shared<SiftGPURegistration>(objectRGBImage, sceneRGBImage,
	//		objectDepthImage, sceneDepthImage);
	//	else{
	//		if (it == sortedHashKeyList.begin())
	//			reg->SetObject(objectRGBImage, objectDepthImage);
	//		reg->SetScene(sceneRGBImage, sceneDepthImage);
	//	}

	//	reg->Apply();
	//	cout << ind << " to #" << it->second << " 3 points method: " << reg->GetInliersNum()
	//		<< " Similirity: " << it->first << endl;

	//	//若3点法内点超过30则直接认为正确
	//	if (reg->GetInliersNum() > 30){
	//		return keyFrameList[it->second].robotPos*reg->GetTransformationMatrix();
	//	}
	//	//以内点个数作为排序依据对原有序列进行重排
	//	s.insert(pair<int, int>(reg->GetInliersNum(), it->second));
	//}

	////若均不符合则使用5点法
	//times = 0;
	//for (auto it = s.begin(); it != s.end() /*&& times < 20*/; it++, times++){
	//	cv::Mat sceneRGBImage = cv::imread(bathPath + keyFrameList[it->second].rgbImageName),
	//		sceneDepthImage = cv::imread(bathPath + keyFrameList[it->second].depthImageName, cv::IMREAD_ANYDEPTH);
	//	reg->SetScene(sceneRGBImage, sceneDepthImage);
	//	reg->Apply();
	//	reg->ApplyWith5Pt();
	//	cout << ind << " to #" << it->second << " 5 points method: " << reg->Get5PointInlierWithDepthNum()
	//		<< '/' << reg->GetInliersNum() << endl;

	//	//cv::Mat img;
	//	//reg->DrawMatch(img);
	//	//char ch[100];
	//	//sprintf(ch, "loction%d_%d.jpg", ind, it->second);
	//	//cv::imwrite(ch, img);

	//	if (reg->Get5PointInlierNum() > 15 && reg->Get5PointInlierWithDepthNum() > 5){
	//		return keyFrameList[it->second].robotPos*reg->GetTransformationMatrix();
	//	}
	//}

	////若5点法也无法得到则使用3点法最多的一帧进行ICP
	//auto it = s.begin();
	//cv::Mat sceneRGBImage = cv::imread(bathPath + keyFrameList[it->second].rgbImageName),
	//	sceneDepthImage = cv::imread(bathPath + keyFrameList[it->second].depthImageName, cv::IMREAD_ANYDEPTH);
	//reg->SetScene(sceneRGBImage, sceneDepthImage);
	//reg->Apply();

	//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
	//	objectPointCloud = Miscellaneous<>::GeneratePointCloud(objectDepthImage, objectRGBImage, 0.01),
	//	scenePointCloud = Miscellaneous<>::GeneratePointCloud(sceneDepthImage, sceneRGBImage, 0.01),
	//	transPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	//pcl::GeneralizedIterativeClosestPoint6D icp;
	//pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
	//icp.setMaxCorrespondenceDistance(0.08);
	//icp.setTransformationEpsilon(1e-10);
	//icp.setEuclideanFitnessEpsilon(0.0005);
	//icp.setMaximumIterations(50);
	//icp.setInputSource(objectPointCloud);
	//icp.setInputTarget(scenePointCloud);
	//icp.align(*transPointCloud, reg->GetTransformationMatrix());
	//cout << "target to #" << it->second << " ICP method: " << endl;
	//return keyFrameList[it->second].robotPos*icp.getFinalTransformation();

	//若无匹配则返回单位矩阵
	return Eigen::Matrix4f::Identity();
}