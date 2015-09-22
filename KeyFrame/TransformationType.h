#ifndef TRANSFORMATION_TYPE_H_
#define TRANSFORMATION_TYPE_H_

#include "common.h"
#include <pcl/registration/transformation_estimation_svd.h>
class TransformationType
{
public:
	enum TransFormat
	{
		rotate,
		translation,
		rigid,
		affine,
		Similarity,
		projection
	};

	template <typename TPoint = pcl::PointXYZ>
	static void rigidEval(typename pcl::PointCloud<TPoint>::Ptr pointCloud1, const vector<int>& indices,
		typename pcl::PointCloud<TPoint>::Ptr pointCloud2, Eigen::MatrixXf& transMatrix)
	{
		pcl::registration::TransformationEstimationSVD<TPoint, TPoint> registest =
			pcl::registration::TransformationEstimationSVD<TPoint, TPoint>(true);
		Eigen::Matrix4f temp;
		boost::shared_ptr<pcl::PointCloud<TPoint>> t1(new pcl::PointCloud<TPoint>),
			t2(new pcl::PointCloud<TPoint>);
		for (auto it = indices.begin(); it != indices.end(); it++)
		{
			t1->push_back(pointCloud1->points[*it]);
			t2->push_back(pointCloud2->points[*it]);
		}
		registest.estimateRigidTransformation(*t1, *t2, temp);
		transMatrix = temp;
	}
	
	template <typename TPoint = pcl::PointXYZ>
	static void translationEval(typename pcl::PointCloud<TPoint>::Ptr pointCloud1, const vector<int>& indices,
		typename pcl::PointCloud<TPoint>::Ptr pointCloud2, Eigen::MatrixXf& transMatrix)
	{
		TPoint t;
		t.x = 0;t.y = 0;t.z = 0;
		for (auto it = indices.begin(); it != indices.end(); it++)
		{
			t.x += pointCloud2->points[*it].x-pointCloud1->points[*it].x;
			t.y += pointCloud2->points[*it].y-pointCloud1->points[*it].y;
			t.z += pointCloud2->points[*it].z-pointCloud1->points[*it].z;
		}
		t.x /= indices.size();
		t.y /= indices.size();
		t.z /= indices.size();
		
		transMatrix = Eigen::Matrix4f::Identity();
		transMatrix(0,3) = t.x;
		transMatrix(1,3) = t.y;
		transMatrix(2,3) = t.z;
	}
};
#endif