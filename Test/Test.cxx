#include "KeyFrameSimple.h"
#include "common.h"
#include "SiftGPURegistration.h"
#include "Miscellaneous.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include <g2o/core/factory.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include "g2o/types/slam3d/types_slam3d.h"
using namespace std;
void loadfile(fstream& file, Eigen::Isometry3d& transf, g2o::OptimizableGraph::Vertex* v1, g2o::OptimizableGraph::Vertex* v2, double norm){
	file << "**********************************" << std::endl;
	file << "Vertex " << v1->id() << " to " << "Vertex " << v2->id() << std::endl;
	file << "Average deviation:" << norm << std::endl;
	g2o::Vector6d vec6D = g2o::internal::toVectorET(transf);
	file << "roll: " << vec6D(3) << "   pitch: " << vec6D(4) << "    yaw: " << vec6D(5) << endl;
	file << "X distance: " << vec6D(0) << "   Y distance: " << vec6D(1) << "   Z distance: " << vec6D(2) << endl;
}

void deleteAloneVertex(g2o::SparseOptimizer& opt){
	//hash table保存主节点ID
	vector<int> mainVertex(1000, false);
	//使用DFS得到主节点
	stack<g2o::OptimizableGraph::Vertex*> s;
	s.push(opt.vertex(0));
	mainVertex[0] = true;
	while (!s.empty()){
		g2o::OptimizableGraph::EdgeSet es = s.top()->edges();
		bool finishFlag = true;
		for (auto it = es.begin(); it != es.end(); it++){
			g2o::OptimizableGraph::Edge* e = (g2o::OptimizableGraph::Edge*)*it;
			g2o::OptimizableGraph::Vertex* v1 = (g2o::OptimizableGraph::Vertex*)e->vertices()[0];
			g2o::OptimizableGraph::Vertex* v2 = (g2o::OptimizableGraph::Vertex*)e->vertices()[1];
			if (!mainVertex[v1->id()]){
				mainVertex[v1->id()] = true;
				s.push(v1);
				finishFlag = false;
				break;
			}
			if (!mainVertex[v2->id()]){
				mainVertex[v2->id()] = true;
				s.push(v2);
				finishFlag = false;
				break;
			}
		}
		if (finishFlag)
			s.pop();
	}
	//去除非主节点
	//经验证同时能去除孤立边
	for (auto it = mainVertex.begin(); it != mainVertex.end();){
		if (!*it && opt.vertex(it - mainVertex.begin()) != NULL){
			cout << "Vertex id: " << it - mainVertex.begin() << endl;
			opt.removeVertex(opt.vertex(it - mainVertex.begin()));
		}
		else
			it++;			
	}
}

int deleteUnstableEdge(g2o::SparseOptimizer& opt){
	g2o::BlockSolverX::LinearSolverType * linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>();
	g2o::BlockSolverX* blockSolver = new g2o::BlockSolverX(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* optimizationAlgorithm = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(true);

	set<int> vertex;
	for (auto it = opt.edges().begin(); it != opt.edges().end(); it++){
		g2o::EdgeSE3* e = (g2o::EdgeSE3*)*it;
		g2o::VertexSE3* v1 = (g2o::VertexSE3*)e->vertices()[0];
		g2o::VertexSE3* v2 = (g2o::VertexSE3*)e->vertices()[1];
		g2o::VertexSE3 *v1n, *v2n;
		g2o::EdgeSE3* en = new g2o::EdgeSE3;
		if (vertex.count(v1->id()) == 0){
			vertex.insert(v1->id());
			v1n = new g2o::VertexSE3;
			v1n->setId(v1->id());
			v1n->setEstimate(g2o::Isometry3D::Identity());
			optimizer.addVertex(v1n);
		}
		if (vertex.count(v2->id()) == 0){
			vertex.insert(v2->id());
			v2n = new g2o::VertexSE3;
			v2n->setId(v2->id());
			v2n->setEstimate(g2o::Isometry3D::Identity());
			optimizer.addVertex(v2n);
		}
		en->vertices()[0] = optimizer.vertex(v1->id());
		en->vertices()[1] = optimizer.vertex(v2->id());
		Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
		information.block<3, 3>(0, 0) *= 400;
		information.block<3, 3>(3, 3) *= 2000;
		en->setInformation(information);
		Eigen::Isometry3d T = e->measurement();
		en->setMeasurement(T);

		g2o::RobustKernel* rb_e = g2o::RobustKernelFactory::instance()->construct("Cauchy");;
		en->setRobustKernel(rb_e);
		optimizer.addEdge(en);
	}

	optimizer.setAlgorithm(optimizationAlgorithm);
	optimizer.initializeOptimization();
	optimizer.optimize(200);

	//Edge validation
	set<pair<int, int>> unstableEdges;
	for (auto it = optimizer.edges().begin(); it != optimizer.edges().end(); it++){
		//for each loop candidate edge
		g2o::OptimizableGraph::Edge* e = (g2o::OptimizableGraph::Edge*)*it;
		g2o::OptimizableGraph::Vertex* v1 = (g2o::OptimizableGraph::Vertex*)e->vertices()[0];
		g2o::OptimizableGraph::Vertex* v2 = (g2o::OptimizableGraph::Vertex*)e->vertices()[1];
		if (v1->id() - 1 == v2->id()) continue;

		g2o::Vector7d m;
		e->getMeasurementData(&m[0]);
		Eigen::Isometry3d transf = g2o::internal::fromVectorQT(m);
		Eigen::Matrix4f tran = Miscellaneous<>::Isometry3dToMatrixXf(transf);

		//Calculate delta x

		//valid edge relation
		g2o::Vector7d _estimate1;
		v1->getEstimateData(&_estimate1(0));
		Eigen::Isometry3d i3d1 = g2o::internal::fromVectorQT(_estimate1);
		Eigen::Matrix4f Pos1 = Miscellaneous<>::Isometry3dToMatrixXf(i3d1);

		g2o::Vector7d _estimate2;
		v2->getEstimateData(&_estimate2(0));
		Eigen::Isometry3d i3d2 = g2o::internal::fromVectorQT(_estimate2);
		Eigen::Matrix4f Pos2 = Miscellaneous<>::Isometry3dToMatrixXf(i3d2);

		Eigen::Matrix4f transPos1 = Pos1*tran;
		Eigen::Isometry3d Pos1_3d = Miscellaneous<>::MatrixXfToIsometry3d(transPos1);
		g2o::Vector6d vec6D1 = g2o::internal::toVectorET(Pos1_3d);

		Eigen::Isometry3d Pos2_3d = Miscellaneous<>::MatrixXfToIsometry3d(Pos2);
		g2o::Vector6d vec6D2 = g2o::internal::toVectorET(Pos2_3d);

		double delta_x = sqrt((vec6D1[0] - vec6D2[0])*(vec6D1[0] - vec6D2[0])
			+ (vec6D1[1] - vec6D2[1])*(vec6D1[1] - vec6D2[1])
			+ (vec6D1[2] - vec6D2[2])*(vec6D1[2] - vec6D2[2]));
		double delta_angle = sqrt((vec6D1[3] - vec6D2[3])*(vec6D1[3] - vec6D2[3])
			+ (vec6D1[4] - vec6D2[4])*(vec6D1[4] - vec6D2[4])
			+ (vec6D1[5] - vec6D2[5])*(vec6D1[5] - vec6D2[5]));
		Eigen::Matrix4f delta = Pos2 - Pos1*tran;
		double deltan = delta.norm();

		std::cout << "**********************************" << std::endl;
		std::cout << "Vertex " << v1->id() << " to " << "Vertex " << v2->id() << std::endl;
		std::cout << "Edge delta: " << deltan << std::endl;
		std::cout << "Translation delta: " << delta_x << std::endl;
		std::cout << "Angle delta: " << delta_angle << std::endl;
		std::cout << "Average deviation:" << deltan << std::endl;
		g2o::Vector6d vec6D = g2o::internal::toVectorET(transf);
		cout << "roll: " << vec6D(3) << "   pitch: " << vec6D(4) << "    yaw: " << vec6D(5) << endl;
		cout << "X distance: " << vec6D(0) << "   Y distance: " << vec6D(1) << "   Z distance: " << vec6D(2) << endl;

		if (delta_x > 0.1){
			cout << "delete edge: " << v1->id() << "<->" << v2->id() << endl;
			unstableEdges.insert(pair<int, int>(v1->id(), v2->id()));
		}			
	}

	vector<g2o::EdgeSE3*> deleteEdge;
	for (auto it = opt.edges().begin(); it != opt.edges().end();it++){
		g2o::EdgeSE3* e = (g2o::EdgeSE3*)*it;
		g2o::VertexSE3* v1 = (g2o::VertexSE3*)e->vertices()[0];
		g2o::VertexSE3* v2 = (g2o::VertexSE3*)e->vertices()[1];
		if (unstableEdges.count(pair<int, int>(v1->id(), v2->id())) > 0){
			cout << "delete edge: " << v1->id() << "<->" << v2->id() << endl;
			deleteEdge.push_back(e);	
		}
	}
	for (auto it = deleteEdge.begin(); it != deleteEdge.end(); it++)
		opt.removeEdge(*it);

	return unstableEdges.size();
}


int main(int argc, char** argv)
{
	g2o::BlockSolverX::LinearSolverType * linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>();
	g2o::BlockSolverX* blockSolver = new g2o::BlockSolverX(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* optimizationAlgorithm = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
	g2o::SparseOptimizer * optimizer = new g2o::SparseOptimizer;

	string bathPath = argv[1];
	optimizer->load(string(bathPath + "BeforeOptimize.g2o").c_str());
	deleteAloneVertex(*optimizer);
	optimizer->save(string(bathPath + "BeforeOptimize_2.g2o").c_str());
	delete optimizer;
	optimizer = new g2o::SparseOptimizer;
	optimizer->load(string(bathPath + "BeforeOptimize_2.g2o").c_str());
	deleteUnstableEdge(*optimizer);

	for (auto it = optimizer->edges().begin(); it != optimizer->edges().end(); it++){
		g2o::EdgeSE3* e = (g2o::EdgeSE3*)*it;
		g2o::RobustKernel* rb_e = g2o::RobustKernelFactory::instance()->construct("Cauchy");;
		e->setRobustKernel(rb_e);
		g2o::VertexSE3* v1 = (g2o::VertexSE3*)e->vertices()[0];
		g2o::VertexSE3* v2 = (g2o::VertexSE3*)e->vertices()[1];
		v1->setEstimate(g2o::Isometry3D::Identity());
		v2->setEstimate(g2o::Isometry3D::Identity());
	}
	optimizer->setVerbose(true);
	optimizer->setAlgorithm(optimizationAlgorithm);
	////reognization add edges
	optimizer->initializeOptimization();
	optimizer->optimize(200);
	

	vector<KeyFrameSimple> keyframeSimpleList;
	KeyFrameSimple::LoadKeyFrameList(keyframeSimpleList, bathPath);
	pcl::PLYWriter plyWrite;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr allPointCloudDownsample(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
	float sampleRatio = 0.03;
	sor.setLeafSize(sampleRatio, sampleRatio, sampleRatio);

	for (auto it = keyframeSimpleList.begin(); it != keyframeSimpleList.end(); it++){
		//if (it - keyframeSimpleList.begin() < 90)
		//	continue;
		//if (it - keyframeSimpleList.begin() > 110)
		//	break;
		g2o::OptimizableGraph::Vertex* v = optimizer->vertex(it - keyframeSimpleList.begin());
		if (v == NULL) continue;
		g2o::Vector7d _estimate;
		v->getEstimateData(&_estimate(0));
		Eigen::Isometry3d i3d = g2o::internal::fromVectorQT(_estimate);
		it->robotPos = Miscellaneous<>::Isometry3dToMatrixXf(i3d);

		Eigen::Matrix4f transformation = keyframeSimpleList.begin()->robotPos.inverse()*it->robotPos;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr allPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr transPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
		cv::Mat depthImage = cv::imread(bathPath + it->depthImageName, cv::IMREAD_ANYDEPTH),
			rgbImage = cv::imread(bathPath + it->rgbImageName);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tempPointCloud =
			Miscellaneous<>::GeneratePointCloud(depthImage, rgbImage, sampleRatio);
		pcl::transformPointCloud(*tempPointCloud, *transPointCloud, transformation);

		//pcl::PLYWriter plyWrite;
		//char name[50];
		//std::sprintf(name, "%dTo%d.ply", it - keyframeSimpleList.begin(), 0);
		//plyWrite.write(bathPath + name, *transPointCloud, true);

		*allPointCloud = *transPointCloud + *allPointCloudDownsample;
		sor.setInputCloud(allPointCloud);
		sor.filter(*allPointCloudDownsample);
		cout << "point cloud size: " << allPointCloudDownsample->size() << endl;
	}
	plyWrite.write(bathPath + "result.ply", *allPointCloudDownsample, true);

	KeyFrameSimple::SaveKeyFrameList(keyframeSimpleList, bathPath);

	delete optimizer;
	return 0;
}

