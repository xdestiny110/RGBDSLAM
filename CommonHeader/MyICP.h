#ifndef MYICP_H_
#define MYICP_H_

#include <common.h>
#include <TransformationType.h>
#include <Miscellaneous.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>

template <typename TPoint>
class MyICP
{
public:
	MyICP() :maxDistance(0.1), maxEps(0.001), type(TransformationType::translation)
	{}
	MyICP& SetSourcePoint(typename const pcl::PointCloud<TPoint>::Ptr _srcPoint){
		srcPoint = _srcPoint;
		return *this;
	}
	MyICP& SetTargetPoint(typename const pcl::PointCloud<TPoint>::Ptr _tgtPoint){
		tgtPoint = _tgtPoint;
		return *this;
	}
	MyICP& SetMaxDistance(double dis){
		maxDistance = dis;
		return *this;
	}
	MyICP& SetMaxEps(double _eps){
		maxEps = _eps;
		return *this;
	}
	MyICP& SetType(TransformationType::TransFormat _type){
		type = _type;
		return *this;
	}
	Eigen::MatrixXf GetTransformation(){
		return transformation;
	}
	MyICP& Align(Eigen::MatrixXf guess = Eigen::MatrixXf::Identity()){
		typename pcl::PointCloud<TPoint>::Ptr src(new pcl::PointCloud<TPoint>),
			tgt(new pcl::PointCloud<TPoint>), trans(new pcl::PointCloud<TPoint>);

		transformation = guess;

		pcl::search::KdTree<TPoint> tree;
		tree.setInputCloud(tgtPoint);
		vector<int> index, indices;
		vector<float> distance;
		double err = 100;

		while (err > maxEps){
			err = 0;
			pcl::transformPointCloud(*srcPoint, *trans, transformation);
			for (auto it = trans->begin(); it != trans->end(); it++){
				tree.nearestKSearch(*it, 1, index, distance);
				if (distance[0] < maxDistance){
					indices.push_back(src->size());
					src->push_back(*it);
					tgt->push_back(tgtPoint->points[index[0]]);
				}
			}
			Eigen::MatrixXf prev = transformation;

			switch (type)
			{
			case TransformationType::rotate:
				break;
			case TransformationType::translation:
				TransformationType::translationEval<TPoint>(src, indices, tgt, transformation);
				transformation = transformation*prev;
				//cout << transformation << endl;
				break;
			case TransformationType::rigid:
				break;
			case TransformationType::affine:
				break;
			case TransformationType::Similarity:
				break;
			case TransformationType::projection:
				break;
			default:
				break;
			}
			err = (transformation - prev).norm();
		}
		return *this;
	}


	
private:
	typename pcl::PointCloud<TPoint>::Ptr srcPoint;
	typename pcl::PointCloud<TPoint>::Ptr tgtPoint;
	double maxDistance, maxEps;
	TransformationType::TransFormat type;
	Eigen::MatrixXf transformation;
};

#endif