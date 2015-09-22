#ifndef GRAPHEND_H_
#define GRAPHEND_H_

#include "common.h"
#include "Miscellaneous.h"
#include "RegistrationBase.h"
#include "KeyFrame.h"
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
class GraphEnd
{
public:
	GraphEnd()
	{

	}
	GraphEnd& GenerateKeyFrame(const KeyFrame&,bool fixed = false);
	GraphEnd& GenerateEdge(int, int, Eigen::MatrixXf, int);
	GraphEnd& GraphEnd::Optimize(int step = 200, bool verbose = true);
	GraphEnd& FindClouse();
	int CheckClouse();
	GraphEnd& SavePLY(const string&, float sampleRatio = 0.01);
	GraphEnd& SaveGraph(const string&);
	GraphEnd& SaveTrajectory(const string&);
	GraphEnd& SaveKeyFrame(const string&);
	cv::Mat currentRGBImage, currentDepthImage, preRGBImage, preDepthImage;
	vector<cv::KeyPoint> preKeyPoint;
	Eigen::MatrixXf preTransformation;
	bool preTrust;
	RegistrationBase::Ptr reg;
	vector<KeyFrame> keyframeVector;

private:	
	g2o::SparseOptimizer opt;
	g2o::OptimizationAlgorithmLevenberg* solver;
	g2o::RobustKernel* robustKernel;
	SlamLinearSolver* linearSolver;
	SlamBlockSolver* blockSolver;
	Eigen::Matrix4f icpCalculate(pcl::PointCloud<pcl::PointXYZ>::Ptr,
		pcl::PointCloud<pcl::PointXYZ>::Ptr, const Eigen::Matrix4f&, bool flag = false, bool = false);
	bool overlapCalc(pcl::PointCloud<pcl::PointXYZ>::Ptr,
		pcl::PointCloud<pcl::PointXYZ>::Ptr, const Eigen::Matrix4f&, float = 0.03, float = 0.75);
	bool planeCalc(pcl::PointCloud<pcl::PointXYZ>::ConstPtr, pcl::PointCloud<pcl::PointXYZ>::ConstPtr, float = 0.8);
};

#endif