#ifndef RANSAC_H_
#define RANSAC_H_

#include "common.h"
#include "TransformationType.h"
#include "Miscellaneous.h"
#include <pcl/common/transforms.h>

template <typename TPoint = pcl::PointXYZRGBA>
class Ransac
{
public:
	Ransac(typename pcl::PointCloud<TPoint>::Ptr pointCloud1,
		typename pcl::PointCloud<TPoint>::Ptr pointCloud2,
		TransformationType::TransFormat type = TransformationType::rigid,
		float _threshold = 0.05, float _confidence = 0.99, int _maxIterNum = 1024)
		:maxIterNum(_maxIterNum), confidence(_confidence), threshold(_threshold)
	{
		switch (type)
		{
		case TransformationType::rigid:
			ransacMain(pointCloud1, pointCloud2, TransformationType::rigidEval<TPoint>, 3);
			break;
		}
	}

	void operator()(typename pcl::PointCloud<TPoint>::Ptr pointCloud1,
		typename pcl::PointCloud<TPoint>::Ptr pointCloud2, 
		TransformationType::TransFormat type = TransformationType::rigid,
		float _threshold = 0.05, float _confidence = 0.99, int _maxIterNum = 1024) const
	{
		switch (type)
		{
		case TransformationType::rigid:
			ransacMain(pointCloud1, pointCloud2, TransformationType::rigidEval<TPoint>, 3);
			break;
		}
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
	using funcEval = void(*)(typename pcl::PointCloud<TPoint>::Ptr, const vector<int>&,
		typename pcl::PointCloud<TPoint>::Ptr, Eigen::MatrixXf&);

	void ransacMain(typename pcl::PointCloud<TPoint>::Ptr pointCloud1,
		typename pcl::PointCloud<TPoint>::Ptr pointCloud2, funcEval func, int sampleNum)
	{
		inliersNum = 0;
		inliersIndex.clear();
		if (pointCloud1->size() < sampleNum || pointCloud2->size() < sampleNum)
		{
			cerr << "no enough points for ransac!" << endl;
			transforMat = Eigen::Matrix4f::Identity();
			return;
		}
		int iterNum = 0;
		//cv::RNG rng(cv::getTickCount());
		cv::RNG rng(1);
		while (iterNum <= maxIterNum)
		{
			//产生随机抽样点
			vector<int> sample;
			for (int i = 0; i < sampleNum; i++)
			{
				int ind = rng.uniform(0.0, (double)(*pointCloud1).points.size());
				while (find(sample.begin(), sample.end(), ind) != sample.end())
				{
					ind = rng.uniform(0.0, (double)(*pointCloud1).points.size());
				}
				sample.push_back(ind);
			}
			Eigen::MatrixXf transTemp;
			func(pointCloud1, sample, pointCloud2, transTemp);

			//计算内点概率
			int inliersNumTemp = 0;
			vector<int> inlierIndexTemp;
			typename pcl::PointCloud<TPoint>::Ptr pointCloudTemp(new pcl::PointCloud<TPoint>);
			pcl::transformPointCloud(*pointCloud1, *pointCloudTemp, transTemp);
			for (int i = 0; i < (*pointCloud1).points.size(); i++)
			{
				Eigen::RowVectorXf v1 = Miscellaneous<TPoint>::PointToRowVector((*pointCloudTemp).points[i]),
					v2 = Miscellaneous<TPoint>::PointToRowVector((*pointCloud2).points[i]);
				if ((v2 - v1).norm() < threshold)
				{
					inliersNumTemp++;
					inlierIndexTemp.push_back(i);
				}
			}

			if (inliersNumTemp>inliersNum)
			{
				inliersIndex.assign(inlierIndexTemp.begin(), inlierIndexTemp.end());
				inliersNum = inliersNumTemp;
				float inliersRate = (float)inliersNum / pointCloud1->points.size();
				maxIterNum = (abs(log(1 - confidence) / log(1 - pow(inliersRate, sampleNum))) > maxIterNum) ?
				maxIterNum : abs(log(1 - confidence) / log(1 - pow(inliersRate, sampleNum)));
			}
			iterNum++;
		}

		func(pointCloud1, inliersIndex, pointCloud2, transforMat);
	}

};
#endif