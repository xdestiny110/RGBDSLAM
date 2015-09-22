#ifndef RANSAC5PT_H_
#define RANSAC5PT_H_

#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

#include "5point.h"


using namespace std;

class Ransac5pt {
public:
	int inlierNumWithDepth;

	Ransac5pt(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud2,
		pcl::PointCloud<pcl::PointXY>::Ptr imgPtCld1, pcl::PointCloud<pcl::PointXY>::Ptr imgPtCld2,
		//Node & lastNode, Node & thisNode, 
		//vector<cv::KeyPoint> & keyPtsLstNode, vector<cv::KeyPoint> & keyPtsThsNode,
		float _threshold = 0.04, float _confidence = 0.99, int _maxIterNum = 2048)
		:maxIterNum(_maxIterNum), confidence(_confidence), threshold(_threshold)
	{
		ransacMain(pointCloud1, pointCloud2, imgPtCld1, imgPtCld2);
	}

	Eigen::MatrixXf GetTransforMat()
	{
		return transforMat;
	}

	vector<int> GetInliersIndex()
	{
		return inliersIndex;
	}

	int GetInliersNum()
	{
		return inliersNum;
	}

private:
	float confidence, threshold;
	int maxIterNum, inliersNum;
	Eigen::MatrixXf transforMat;
	vector<int> inliersIndex;

	void ransacMain(
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud1,
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud2,
		pcl::PointCloud<pcl::PointXY>::Ptr imgPtCld1,
		pcl::PointCloud<pcl::PointXY>::Ptr imgPtCld2);
};

#endif
