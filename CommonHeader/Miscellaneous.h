#ifndef MISCELLANEOUS_H_
#define MISCELLANEOUS_H_

#include "common.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

template <typename TPoint = pcl::PointXYZRGBA>
class Miscellaneous
{
public:
	static const float zMax;
	static cv::Mat PointCloudToMat(boost::shared_ptr<pcl::PointCloud<TPoint>> pointCloud)
	{
		cv::Mat result;
		auto f = bind([&result](TPoint& point)
		{
			cv::Mat M = (cv::Mat_<float>(1, 3) << point.x, point.y, point.z);
			result.push_back(M);
		}, placeholders::_1);
		for_each((*pointCloud).points.begin(), (*pointCloud).points.end(), f);
		return result;
	}

	static Eigen::MatrixXf PointCloudToMatrix(boost::shared_ptr<pcl::PointCloud<TPoint>> pointCloud)
	{
		Eigen::MatrixXf result((*pointCloud).points.size(), 3);
		int i = 0;

		auto f = bind([&](TPoint& point)
		{
			result.row(i) << point.x, point.y, point.z;
			i++;
		}, placeholders::_1);
		for_each((*pointCloud).points.begin(), (*pointCloud).points.end(), f);
		return result;
	}

	static Eigen::VectorXf PointToVector(TPoint& point)
	{
		Eigen::VectorXf result(3);
		result << point.x, point.y, point.z;
		return result;
	}

	static Eigen::RowVectorXf PointToRowVector(TPoint& point)
	{
		Eigen::RowVectorXf result(3);
		result << point.x, point.y, point.z;
		return result;
	}

	static cv::KeyPoint Point3DTo2D(const TPoint& point3D,
		float focalLength = 620.0, float centerX = 319.5, float centerY = 239.5, float scalingFactor = 1000.0)
	{
		cv::KeyPoint kpt;
		kpt.pt.x = point3D.x*focalLength / point3D.z + centerX;
		kpt.pt.y = point3D.y*focalLength / point3D.z + centerY;
		return kpt;
	}

	static TPoint Point2DTo3D(const cv::KeyPoint& point2D,uint16_t idensity,
		float focalLength = 620.0, float centerX = 319.5, float centerY = 239.5, float scalingFactor = 1000.0)
	{
		TPoint pt;
		pt.z = idensity / scalingFactor;
		pt.x = (point2D.pt.x - centerX)*pt.z / focalLength;
		pt.y = (point2D.pt.y - centerY)*pt.z / focalLength;
		return pt;
	}

	static typename pcl::PointCloud<TPoint>::Ptr GeneratePointCloud(const cv::Mat& img,
		float leafSize = 0.0)
	{
		typename pcl::PointCloud<TPoint>::Ptr ptCloud(new pcl::PointCloud<TPoint>);

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				uint16_t identity = img.at<uint16_t>(i, j);
				if (identity <= 0.000001) continue;
				cv::KeyPoint kpt;
				kpt.pt.x = j; kpt.pt.y = i;
				TPoint tpt = Point2DTo3D(kpt, identity);
				if (tpt.z > zMax) continue;
				ptCloud->push_back(tpt);
			}
		}

		if (abs(leafSize) > 0.001)
		{
			typename pcl::PointCloud<TPoint>::Ptr ptSampleCloud(new pcl::PointCloud<TPoint>);
			pcl::VoxelGrid<TPoint> sor;
			sor.setLeafSize(leafSize, leafSize, leafSize);
			sor.setInputCloud(ptCloud);
			sor.filter(*ptSampleCloud);

			typename pcl::PointCloud<TPoint>::Ptr ptFilterCloud(new pcl::PointCloud<TPoint>);
			pcl::StatisticalOutlierRemoval<TPoint> filter;
			filter.setInputCloud(ptSampleCloud);
			filter.setMeanK(50);
			filter.setStddevMulThresh(1.0);
			filter.filter(*ptFilterCloud);

			return ptFilterCloud;
		}

		return ptCloud;
	}

	static typename pcl::PointCloud<TPoint>::Ptr GeneratePointCloud(const cv::Mat& depthImg,
		const cv::Mat& rgbImg, float leafSize = 0.0)
	{
		pcl::PointCloud<TPoint>::Ptr ptCloud(new pcl::PointCloud<TPoint>);
		for (int i = 0; i < depthImg.rows; i++)
		{
			for (int j = 0; j < depthImg.cols; j++)
			{
				uint16_t identity = depthImg.at<uint16_t>(i, j);
				if (identity <= 0.000001) continue;
				cv::KeyPoint kpt;
				kpt.pt.x = j; kpt.pt.y = i;
				TPoint tpt = Point2DTo3D(kpt, identity);
				if (tpt.z > zMax) continue;
				ptCloud->push_back(tpt);
				ptCloud->back().b = rgbImg.at<cv::Vec3b>(i, j)[0];
				ptCloud->back().g = rgbImg.at<cv::Vec3b>(i, j)[1];
				ptCloud->back().r = rgbImg.at<cv::Vec3b>(i, j)[2];
			}
		}

		if (abs(leafSize) > 0.001)
		{		
			typename pcl::PointCloud<TPoint>::Ptr ptSampleCloud(new pcl::PointCloud<TPoint>);
			pcl::VoxelGrid<TPoint> sor;
			sor.setLeafSize(leafSize, leafSize, leafSize);
			sor.setInputCloud(ptCloud);
			sor.filter(*ptSampleCloud);

			typename pcl::PointCloud<TPoint>::Ptr ptFilterCloud(new pcl::PointCloud<TPoint>);
			pcl::StatisticalOutlierRemoval<TPoint> filter;
			filter.setInputCloud(ptSampleCloud);
			filter.setMeanK(50);
			filter.setStddevMulThresh(1.0);
			filter.filter(*ptFilterCloud);

			return ptFilterCloud;
		}


		return ptCloud;
	}

	static void GenerateRangeImage(const string& rangeImageName)
	{
		typename pcl::PointCloud<TPoint>::Ptr pointcloud = GeneratePointCloud(rangeImageName);
		typename pcl::PointCloud<TPoint>::Ptr ptFilterCloud(new pcl::PointCloud<TPoint>);
		pcl::StatisticalOutlierRemoval<TPoint> filter;
		filter.setInputCloud(pointcloud);
		filter.setMeanK(50);
		filter.setStddevMulThresh(1.0);
		filter.filter(*ptFilterCloud);
		cv::Mat depthImg(480, 640, CV_16U, cv::Scalar(0));
		for (auto it = ptFilterCloud->begin(); it != ptFilterCloud->end(); it++){
			cv::KeyPoint kpt = Point3DTo2D(*it);
			depthImg.at<u_int16_t>(kpt.pt) = it->z * 1000;
		}
		cv::imwrite(rangeImageName, depthImg);
	}

	static Eigen::Isometry3d MatrixXfToIsometry3d(Eigen::MatrixXf transformation)
	{
		Eigen::Isometry3d T;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				T(i, j) = transformation(i, j);
			}
		}
		T.makeAffine();
		return T;
	}

	static Eigen::MatrixXf Isometry3dToMatrixXf(Eigen::Isometry3d& transformation)
	{
		Eigen::MatrixXf T(4, 4);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				T(i, j) = transformation(i, j);
			}
		}
		T.block<1, 4>(3, 0) << 0, 0, 0, 1;
		return T;
	}

	static void ShowCloud(typename const pcl::PointCloud<TPoint>::Ptr points1,
		typename const pcl::PointCloud<TPoint>::Ptr points2)
	{
		pcl::visualization::PCLVisualizer viewer;
		viewer.addPointCloud(points1, pcl::visualization::PointCloudColorHandlerCustom<TPoint>(points1, 255, 0, 0), "cloud1");
		viewer.addPointCloud(points2, pcl::visualization::PointCloudColorHandlerCustom<TPoint>(points2, 0, 255, 0), "cloud2");
		viewer.spin();
	};

	static void SaveCloud(const string& path, typename const pcl::PointCloud<TPoint>::Ptr cloud,
		bool bin = true)
	{
		pcl::PLYWriter writer;
		writer.write(path, *cloud, bin);
	}
};

class TimeScope
{
public:
	TimeScope(const string& _title)
		:title(_title)
	{
		timeStart = (double)cv::getTickCount();
	}
	~TimeScope()
	{
		timeFinish = (double)cv::getTickCount();
		cout << title << " time cost: " << (timeFinish - timeStart) / cv::getTickFrequency() << endl;
	}
private:
	double timeStart, timeFinish;
	string title;
};

template <typename TPoint = pcl::PointXYZRGBA>
const float Miscellaneous<TPoint>::zMax = 2.5;
#endif