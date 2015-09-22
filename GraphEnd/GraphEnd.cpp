#include "GraphEnd.h"
#include "ColorHistgram.h"
#include "SiftGPURegistration.h"
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <boost/filesystem.hpp>

GraphEnd& GraphEnd::GenerateKeyFrame(const KeyFrame& kf,bool fixed)
{
	cout << "Generate a new keyframe #" << kf.imgId << endl;
	keyframeVector.push_back(kf);
	keyframeVector.back().hist = ColorHistCalc(kf.rgbImage);

	//建立G2O节点
	g2o::VertexSE3* v = new g2o::VertexSE3;
	v->setId(kf.imgId);
	v->setEstimate(Miscellaneous<>::MatrixXfToIsometry3d(kf.robotPos));
	v->setFixed(fixed);
	opt.addVertex(v);

	//输出单个变换点云
	char name[50];
	std::sprintf(name, "%dTo%d.ply", keyframeVector.back().imgId, 0);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr trans(new pcl::PointCloud<pcl::PointXYZRGBA>),
		objectPointCloud = 
		Miscellaneous<pcl::PointXYZRGBA>::GeneratePointCloud(keyframeVector.back().depthImage, keyframeVector.back().rgbImage, 0.01);
	Eigen::Matrix4f transformation = keyframeVector.begin()->robotPos.inverse()*keyframeVector.back().robotPos;
	pcl::transformPointCloud(*objectPointCloud, *trans, transformation);
	Miscellaneous<>::SaveCloud(name, trans);

	//输出关键帧点云与图像
	//sprintf(name, "keyframe%d.ply", keyframeVector.back().imgId);
	//Miscellaneous<>::SaveCloud(name, keyframeVector.back().pointCloud);

	return *this;
}

GraphEnd& GraphEnd::GenerateEdge(int id1, int id2, Eigen::MatrixXf transformation, int inliersNum)
{
	g2o::EdgeSE3* e = new g2o::EdgeSE3;
	e->vertices()[0] = opt.vertex(id1);
	e->vertices()[1] = opt.vertex(id2);
		

	Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
	information.block<3, 3>(0, 0) *= 400;
	information.block<3, 3>(3, 3) *= 2000;
	e->setInformation(information);
	Eigen::Isometry3d T;
	T = Miscellaneous<>::MatrixXfToIsometry3d(transformation.inverse());
	//T = Miscellaneous<>::MatrixXfToIsometry3d(transformation);
	e->setMeasurement(T);
	g2o::RobustKernel* rb_e = g2o::RobustKernelFactory::instance()->construct("Cauchy");;
	e->setRobustKernel(rb_e);
	opt.addEdge(e);
	cout << "Generate new edge from #" << id1 << " to #" << id2 << endl;
	cout << "Inliers num: " << inliersNum << endl;
	cout << T.matrix() << endl;

	cv::Mat matchImg;
	reg->DrawMatch(matchImg);
	char ch[100];
	sprintf_s(ch, sizeof(ch), "match%dTo%d.jpg",
		id1, id2);
	cv::imwrite(ch, matchImg);

	return *this;
}

GraphEnd& GraphEnd::Optimize(int step, bool verbose)
{
	//hash table保存主节点ID
	vector<bool> mainVertex(opt.vertices().size(), false);
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
	for (auto it = mainVertex.begin(); it != mainVertex.end(); it++){
		if (!*it)
			opt.removeVertex(opt.vertex(it - mainVertex.begin()));
	}
	//去除非主边
	for (auto it = opt.edges().begin(); it != opt.edges().end();){
		g2o::OptimizableGraph::Edge* e = (g2o::OptimizableGraph::Edge*)*it;
		g2o::OptimizableGraph::Vertex* v1 = (g2o::OptimizableGraph::Vertex*)e->vertices()[0];
		g2o::OptimizableGraph::Vertex* v2 = (g2o::OptimizableGraph::Vertex*)e->vertices()[1];
		if (!mainVertex[v1->id()] || !mainVertex[v2->id()])
			opt.removeEdge(e);
		else
			it++;
	}

	linearSolver = new SlamLinearSolver;
	linearSolver->setBlockOrdering(false);
	blockSolver = new SlamBlockSolver(linearSolver);
	solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
	opt.setAlgorithm(solver);
	opt.setVerbose(verbose);
	opt.initializeOptimization();
	opt.optimize(step);

	for (auto it = keyframeVector.begin(); it != keyframeVector.end(); it++)
	{
		g2o::OptimizableGraph::Vertex* v = opt.vertex(it - keyframeVector.begin());
		if (v == NULL) continue;
		g2o::Vector7d _estimate;
		v->getEstimateData(&_estimate(0));
		Eigen::Isometry3d i3d = g2o::internal::fromVectorQT(_estimate);
		it->robotPos = Miscellaneous<>::Isometry3dToMatrixXf(i3d);
	}

	return *this;
}

int GraphEnd::CheckClouse()
{
	//返回值：
	//1:转移矩阵可信且不需建立关键帧
	//2:可信且需建立关键帧
	//3:不可信且不需建立
	//4:不可信且需建立
	//5:转移矩阵错误
	if (reg.use_count() == 0)
		reg = make_shared<SiftGPURegistration>(currentRGBImage, keyframeVector.back().rgbImage,
		currentDepthImage, keyframeVector.back().depthImage);
	else{
		reg->SetObject(currentRGBImage, currentDepthImage);
		reg->SetScene(keyframeVector.back().rgbImage, keyframeVector.back().depthImage,
			keyframeVector.back().keypoint, cv::Mat());
	}
	reg->Apply();
	cout << "inliers num: " << reg->GetInliersNum() << endl;

	g2o::Isometry3D iso3D = Miscellaneous<>::MatrixXfToIsometry3d(reg->GetTransformationMatrix().inverse());
	g2o::Vector6d vec6D = g2o::internal::toVectorET(iso3D);
	cout << "roll: " << vec6D(3) << "   pitch: " << vec6D(4) << "    yaw: " << vec6D(5) << endl;
	cout << "X distance: " << vec6D(0) << "   Y distance: " << vec6D(1) << "   Z distance: " << vec6D(2) << endl;

	if (abs(vec6D(4)) > 0.1 || abs(vec6D(5)) > 0.1 || abs(vec6D(3)) > 0.1 ||
		hypot(hypot(vec6D(0), vec6D(2)), vec6D(1)) > 0.15 || reg->GetInliersNum() < 25){
		pcl::PointCloud<pcl::PointXYZ>::Ptr objectPointCloud, scenePointCloud;
		objectPointCloud =
			Miscellaneous<pcl::PointXYZ>::GeneratePointCloud(currentDepthImage, 0.01);
		scenePointCloud =
			Miscellaneous<pcl::PointXYZ>::GeneratePointCloud(keyframeVector.back().depthImage, 0.01);
		Eigen::Matrix4f icpTransformation;
		//若内点对大于25且重复度达到标准
		if (reg->GetInliersNum() > 25 && overlapCalc(objectPointCloud, scenePointCloud, reg->GetTransformationMatrix()))
		{
			return 2;
		}
		//否则使用5点法
		else
		{
			reg->ApplyWith5Pt();
			cout << "2D point pair num: " << reg->Get5PointInlierNum() << endl;
			cout << "2D point pair num with depth: " << reg->Get5PointInlierWithDepthNum() << endl;
			cout << "5 points transformation matrix :" << endl << reg->GetTransformationMatrix() << endl;

			//g2o::Vector7d vec7d =
			//	g2o::internal::toVectorQT(Miscellaneous<>::MatrixXfToIsometry3d(reg->GetTransformationMatrix()).inverse());
			//cerr << vec7d << endl;

			icpTransformation = reg->GetTransformationMatrix();
			bool g2oFlag = false, icpFlag = false, planeFlag = planeCalc(objectPointCloud, scenePointCloud);
			//若5点法内点大于25且有深度的点对大于15则认为正确
			if (reg->Get5PointInlierNum() > 25 && reg->Get5PointInlierWithDepthNum() > 15
				&& overlapCalc(objectPointCloud, scenePointCloud, icpTransformation)){

			}
			else{
				iso3D = Miscellaneous<>::MatrixXfToIsometry3d(reg->GetTransformationMatrix());
				vec6D = g2o::internal::toVectorET(iso3D);
				cout << "roll: " << vec6D(3) << "   pitch: " << vec6D(4) << "    yaw: " << vec6D(5) << endl;
				cout << "X distance: " << vec6D(0) << "   Y distance: " << vec6D(1) << "   Z distance: " << vec6D(2) << endl;
				//若转移矩阵并非太过于荒谬
				//5点法可以求解且平面占主要成分 或 5点法认为较为可靠
				//则使用小ICP
				if (((reg->Get5PointInlierNum() > 5 && reg->Get5PointInlierWithDepthNum() > 5) ||
					(planeFlag && reg->Get5PointInlierNum() != 0 && reg->Get5PointInlierWithDepthNum() != 0))
					&& abs(vec6D(3)) < 0.35 && abs(vec6D(5)) < 0.35 && abs(vec6D(4)) < 0.35 &&
					hypot(hypot(vec6D(0), vec6D(2)), vec6D(1)) < 0.3){
					icpTransformation = icpCalculate(objectPointCloud, scenePointCloud, icpTransformation);
				}
				//使用大ICP
				else{
					icpFlag = true;
					//判断景物是否以平面为主
					if (reg->Get5PointInlierWithDepthNum() < 5 && planeCalc(objectPointCloud, scenePointCloud)
						&& objectPointCloud->size()<10000){
						return 5;
					}
					else{
						if (reg->Get5PointInlierNum()>25 &&
							abs(vec6D(3)) < 0.35 && abs(vec6D(5)) < 0.35 && abs(vec6D(4)) < 0.35){
							icpTransformation(0, 3) = 0;
							icpTransformation(1, 3) = 0;
							icpTransformation(2, 3) = 0;
							cout << "lock rotation!Using ICP method" << endl;
							icpTransformation = icpCalculate(objectPointCloud, scenePointCloud, icpTransformation, true, true);
						}
						else{
							icpTransformation = Eigen::Matrix4f::Identity();
							cout << "5 points method bug!Using ICP method" << endl;
							icpTransformation = icpCalculate(objectPointCloud, scenePointCloud, icpTransformation, true);
						}
					}					
				}
			}

			reg->SetTransformationMatrix(icpTransformation);
			iso3D = Miscellaneous<>::MatrixXfToIsometry3d(icpTransformation);
			vec6D = g2o::internal::toVectorET(iso3D);
			cout << icpTransformation << endl;
			cout << "roll: " << vec6D(3) << "   pitch: " << vec6D(4) << "    yaw: " << vec6D(5) << endl;
			cout << "X distance: " << vec6D(0) << "   Y distance: " << vec6D(1) << "   Z distance: " << vec6D(2) << endl;
			if (abs(vec6D(4)) < 0.1 && abs(vec6D(5)) < 0.1 && abs(vec6D(3)) < 0.1 &&
				hypot(hypot(vec6D(0), vec6D(2)), vec6D(1)) < 0.15)
				return (icpFlag) ? 3 : 1;
				

			if (abs(vec6D(3)) > 0.35 || abs(vec6D(5)) > 0.35 || abs(vec6D(4)) > 0.35 ||
				hypot(hypot(vec6D(0), vec6D(2)), vec6D(1)) > 0.35)
				return 5;
			
			return (icpFlag) ? 4 : 2;
		}
	}
	else
		return 1;

	return 0;
}

GraphEnd& GraphEnd::FindClouse()
{
	//若总关键帧小于4则不检测闭环
	if (keyframeVector.size() < 5) return *this;
	//进行闭环检测
	//需进行闭环检测的关键帧数量
	const int N = 3, L = 10, K = 10;
	//先与前三帧进行匹配
	//for (int i = 0; i < N; i++)
	//{
	//	auto it = keyframeVector.rbegin() + i + 2;
	//	reg->SetScene(it->rgbImage, it->depthImage, it->keypoint, cv::Mat()).Apply();
	//	cout << "Image #" << keyframeVector.rbegin()->imgId << " to image #" << it->imgId << " inliers numbers: "
	//		<< reg->GetInliersNum() << endl;
	//	if (reg->GetInliersNum()>25)
	//	{
	//		GenerateEdge(keyframeVector.rbegin()->imgId, it->imgId,
	//			reg->GetTransformationMatrix(), reg->GetInliersNum());
	//		cv::Mat matchImg;
	//		reg->DrawMatch(matchImg);
	//		char ch[100];
	//		sprintf_s(ch, sizeof(ch), "match%dTo%d.jpg",
	//			keyframeVector.rbegin()->imgId, it->imgId);
	//		cv::imwrite(ch, matchImg);
	//	}
	//}

	g2o::Isometry3D iso3D;
	g2o::Vector6d vec6D;
	vector<int> flagVec(keyframeVector.size() - 2 - N);
	for (auto it = flagVec.begin(); it != flagVec.end(); it++)
		*it = it - flagVec.begin();
	//依据图像相似度进行闭环检测
	set<pair<float, int>> hashTree;
	for (int i = 0; i < keyframeVector.size() - 2 - N; i++){
		//hashTree.insert(pair<float, int>(ColorHistSimilarity(keyframeVector.back().hashKey,
		//	keyframeVector[flagVec[i]].hashKey), i));
		hashTree.insert(pair<double, int>(1 - cv::compareHist(keyframeVector.back().hist, keyframeVector[flagVec[i]].hist, CV_COMP_CORREL), i));
	}
	int sampleClouse = (flagVec.size() > L) ? L : flagVec.size();
	if (sampleClouse <= 0) return *this;
	int number = 0;
	for (auto it = hashTree.begin(); it != hashTree.end() && number<L; it++, number++){
		reg->SetScene(keyframeVector[flagVec[it->second]].rgbImage, keyframeVector[flagVec[it->second]].depthImage,
			keyframeVector[flagVec[it->second]].keypoint, cv::Mat());
		reg->Apply();
		cout << "image #" << keyframeVector.rbegin()->imgId << " to #" << keyframeVector[flagVec[it->second]].imgId
			<< " inliers Number: " << reg->GetInliersNum() << "  Similarity: " << it->first << endl;
		if (reg->GetInliersNum() >= 25)// && reg->GetRobustFlag())
		{
			GenerateEdge(keyframeVector.rbegin()->imgId, keyframeVector[it->second].imgId,
				reg->GetTransformationMatrix(), reg->GetInliersNum());

			char ch[100];
			//sprintf_s(ch, sizeof(ch), "match%dTo%d.jpg",
			//cv::Mat matchImg;
			//reg->DrawMatch(matchImg);
			//	keyframeVector.rbegin()->imgId, keyframeVector[flagVec[it->second]].imgId);
			//cv::imwrite(ch, matchImg);
			//iso3D = Miscellaneous<>::MatrixXfToIsometry3d(reg->GetTransformationMatrix());
			//vec6D = g2o::internal::toVectorET(iso3D);
			//cout << reg->GetTransformationMatrix() << endl;
			//cout << "roll: " << vec6D(3) << "   pitch: " << vec6D(4) << "    yaw: " << vec6D(5) << endl;
			//cout << "X distance: " << vec6D(0) << "   Y distance: " << vec6D(1) << "   Z distance: " << vec6D(2) << endl;

			sprintf(ch, "match%dTo%d_%d.ply", 
				keyframeVector.rbegin()->imgId, keyframeVector[flagVec[it->second]].imgId, keyframeVector.rbegin()->imgId);
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
				objectPointCloud = Miscellaneous<>::GeneratePointCloud(keyframeVector.rbegin()->depthImage,
				keyframeVector.rbegin()->rgbImage, 0.01),
				scenePointCloud = Miscellaneous<>::GeneratePointCloud(keyframeVector[flagVec[it->second]].depthImage,
				keyframeVector[flagVec[it->second]].rgbImage, 0.01),
				transPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			pcl::transformPointCloud(*objectPointCloud, *transPointCloud, reg->GetTransformationMatrix());
			pcl::PLYWriter writer;
			writer.write(ch, *transPointCloud, true);
			sprintf(ch, "match%dTo%d_%d.ply",
				keyframeVector.rbegin()->imgId, keyframeVector[flagVec[it->second]].imgId, keyframeVector[flagVec[it->second]].imgId);
			writer.write(ch, *scenePointCloud, true);

		}
		flagVec[it->second] = -1;
	}
	for (auto it = flagVec.begin(); it != flagVec.end();)
	{
		if (*it<0)
			it = flagVec.erase(it);
		else
			it++;
	}

	//随机选取关键帧进行闭环检测
	sampleClouse = (flagVec.size() > K) ? K : flagVec.size();
	if (sampleClouse <= 0) return *this;
	cv::RNG rng(cv::getTickCount());
	for (int i = 0; i < sampleClouse; i++)
	{
		int ind = round(rng.uniform(0.6, flagVec.size() + 0.4));
		if (ind <= 0 || ind>flagVec.size())
		{
			cerr << "random seeds error!" << endl;
			continue;
		}
		reg->SetScene(keyframeVector[flagVec[ind - 1]].rgbImage, keyframeVector[flagVec[ind - 1]].depthImage,
			keyframeVector[flagVec[ind - 1]].keypoint, cv::Mat());
		reg->Apply();
		cout << "image #" << keyframeVector.rbegin()->imgId << " to #" << keyframeVector[flagVec[ind - 1]].imgId
			<< " inliers Number: " << reg->GetInliersNum() << endl;

		if (reg->GetInliersNum() >= 25)// && reg->GetRobustFlag())
		{
			GenerateEdge(keyframeVector.rbegin()->imgId, keyframeVector[flagVec[ind - 1]].imgId,
				reg->GetTransformationMatrix(), reg->GetInliersNum());
			cv::Mat matchImg;
			reg->DrawMatch(matchImg);

			char ch[100];
			//sprintf_s(ch, sizeof(ch), "match%dTo%d.jpg",
			//	keyframeVector.rbegin()->imgId, keyframeVector[flagVec[ind - 1]].imgId);
			//cv::imwrite(ch, matchImg);
			//iso3D = Miscellaneous<>::MatrixXfToIsometry3d(reg->GetTransformationMatrix());
			//vec6D = g2o::internal::toVectorET(iso3D);
			//cout << reg->GetTransformationMatrix() << endl;
			//cout << "roll: " << vec6D(3) << "   pitch: " << vec6D(4) << "    yaw: " << vec6D(5) << endl;
			//cout << "X distance: " << vec6D(0) << "   Y distance: " << vec6D(1) << "   Z distance: " << vec6D(2) << endl;

			sprintf(ch, "match%dTo%d_%d.ply",
				keyframeVector.rbegin()->imgId, keyframeVector[flagVec[ind - 1]].imgId, keyframeVector.rbegin()->imgId);
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
				objectPointCloud = Miscellaneous<>::GeneratePointCloud(keyframeVector.rbegin()->depthImage,
				keyframeVector.rbegin()->rgbImage, 0.01),
				scenePointCloud = Miscellaneous<>::GeneratePointCloud(keyframeVector[flagVec[ind - 1]].depthImage,
				keyframeVector[flagVec[ind - 1]].rgbImage, 0.01),
				transPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			pcl::transformPointCloud(*objectPointCloud, *transPointCloud, reg->GetTransformationMatrix());
			pcl::PLYWriter writer;
			writer.write(ch, *transPointCloud, true);
			sprintf(ch, "match%dTo%d_%d.ply",
				keyframeVector.rbegin()->imgId, keyframeVector[flagVec[ind - 1]].imgId, keyframeVector[flagVec[ind - 1]].imgId);
			writer.write(ch, *scenePointCloud, true);
		}
		flagVec.erase(flagVec.begin() + ind - 1);
	}

	return *this;
}

GraphEnd& GraphEnd::SavePLY(const string& path,float sampleRatio)
{
	pcl::PLYWriter plyWrite;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr allPointCloudDownsample(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
	sor.setLeafSize(sampleRatio, sampleRatio, sampleRatio);

	for (auto it = keyframeVector.begin(); it != keyframeVector.end(); it++)
	{
		Eigen::Matrix4f transformation = keyframeVector.begin()->robotPos.inverse()*it->robotPos;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr allPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr transPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tempPointCloud =
			Miscellaneous<>::GeneratePointCloud(it->depthImage, it->rgbImage, 0.01);
		pcl::transformPointCloud(*tempPointCloud, *transPointCloud, transformation);

		*allPointCloud = *transPointCloud + *allPointCloudDownsample;
		sor.setInputCloud(allPointCloud);
		sor.filter(*allPointCloudDownsample);
		cout << "point cloud size: " << allPointCloudDownsample->size() << endl;
	}
	plyWrite.write(path, *allPointCloudDownsample, true);

	return *this;
}

GraphEnd& GraphEnd::SaveGraph(const string& path)
{
	//boost::filesystem::path p = path;
	//boost::filesystem::path rootPath;
	//for (auto it = p.begin(); it != p.end() - 1; it++){
	//	rootPath += *it;
	//	rootPath += "/";
	//}
	//p = rootPath;
	//if (!boost::filesystem::is_directory(p))
	//	boost::filesystem::create_directory(p);

	opt.save(path.c_str());
	return *this;
}

GraphEnd& GraphEnd::SaveTrajectory(const string& path)
{
	//boost::filesystem::path p = path;
	//boost::filesystem::path rootPath;
	//for (auto it = p.begin(); it != p.end() - 1; it++){
	//	rootPath += *it;
	//	rootPath += "/";
	//}
	//p = rootPath;
	//if (!boost::filesystem::is_directory(p))
	//	boost::filesystem::create_directory(p);

	ofstream os(path, ios::out);
	for (auto it = keyframeVector.begin(); it != keyframeVector.end(); it++)
	{
		g2o::OptimizableGraph::Vertex* v = opt.vertex(it->imgId);
		g2o::Vector7d _estimate;
		v->getEstimateData(&_estimate(0));
		os << it->imgId;
		g2o::Isometry3D T = g2o::internal::fromVectorQT(_estimate);
		g2o::Vector6d ET = g2o::internal::toVectorET(T);
		for (int i = 0; i < 6; i++)
		{
			os << ' ' << ET(i);
		}
		os << endl;
	}
	os.close();
	return *this;
}

GraphEnd& GraphEnd::SaveKeyFrame(const string& path)
{
	KeyFrame::SaveKeyFrameList(keyframeVector, path);
	return *this;
}

Eigen::Matrix4f GraphEnd::icpCalculate(pcl::PointCloud<pcl::PointXYZ>::Ptr objectPointCloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr scenePointCloud, const Eigen::Matrix4f& initPos, bool flag, bool lockR)
{
	//pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	//tree1->setInputCloud(objectPointCloud);
	//pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	//tree2->setInputCloud(scenePointCloud);

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputTarget(scenePointCloud);
	//icp.setSearchMethodSource(tree1);
	//icp.setSearchMethodTarget(tree2);
	if (flag)
		icp.setMaxCorrespondenceDistance(0.25);
	else
		icp.setMaxCorrespondenceDistance(0.08);
	icp.setTransformationEpsilon(1e-10);
	icp.setEuclideanFitnessEpsilon(0.0005);
	if (flag)
		icp.setMaximumIterations(50);
	else
		icp.setMaximumIterations(5);
	pcl::PointCloud<pcl::PointXYZ>::Ptr regPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity();
	pcl::transformPointCloud(*objectPointCloud, *regPointCloud, initPos);

	pcl::PLYWriter plywriter;
	if (lockR){
		//	icp.setMaxCorrespondenceDistance(0.1);
		//plywriter.write("object.ply", *objectPointCloud, true);
		//plywriter.write("init.ply", *regPointCloud, true);
		//plywriter.write("scene.ply", *scenePointCloud);
		//	objectPointCloud = regPointCloud;
		//	icp.setInputSource(objectPointCloud);
		//	icp.align(*regPointCloud);
		//	Ti = icp.getFinalTransformation();
		//	Ti.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
		//	pcl::transformPointCloud(*objectPointCloud, *regPointCloud, Ti);
		//	plywriter.write("times.ply", *regPointCloud, true);
		//	return Ti*initPos;
	}

	int turns = 5;
	if (flag) turns = 16;
	for (int i = 0; i < turns; i++){
		PCL_INFO("Iteration Nr. %d.\n", i);
		objectPointCloud = regPointCloud;
		icp.setInputSource(objectPointCloud);
		icp.align(*regPointCloud);
		Ti = icp.getFinalTransformation() * Ti;
		icp.setMaxCorrespondenceDistance(icp.getMaxCorrespondenceDistance() - 0.01);
		//if (lockR){			
		//	plywriter.write("trans.ply", *regPointCloud, true);
		//	objectPointCloud = regPointCloud;
		//	icp.setInputSource(objectPointCloud);
		//	icp.align(*regPointCloud);
		//	Ti.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
		//	Eigen::Matrix4f tt = icp.getFinalTransformation();
		//	cerr << tt << endl;
		//	tt.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
		//	pcl::transformPointCloud(*objectPointCloud, *regPointCloud, tt);
		//}
	}

	return Ti*initPos;
}

bool GraphEnd::overlapCalc(pcl::PointCloud<pcl::PointXYZ>::Ptr objectPointCloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr scenePointCloud, const Eigen::Matrix4f& T, float disTh, float rateTh){
	pcl::PointCloud<pcl::PointXYZ>::Ptr trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*objectPointCloud, *trans, T);
	if (trans->size() > scenePointCloud->size()){
		pcl::PointCloud<pcl::PointXYZ>::Ptr t = trans;
		trans = scenePointCloud;
		scenePointCloud = t;
	}
		

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(scenePointCloud);
	int number = 0;
	for (auto it = trans->begin(); it != trans->end(); it++){
		vector<int> indices;
		vector<float> distance;
		tree->nearestKSearch(*it, 1, indices, distance);
		if (sqrt(distance[0]) < disTh)
			number++;
	}

	float rate = static_cast<float>(number)* 2 / (trans->size() + scenePointCloud->size());
	cout << "pointcloud overlap rate: " << rate << endl;
	return (rate>rateTh);
}

bool GraphEnd::planeCalc(pcl::PointCloud<pcl::PointXYZ>::ConstPtr objectPointCloud,
	pcl::PointCloud<pcl::PointXYZ>::ConstPtr scenePointCloud, float th)
{
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setInputCloud(objectPointCloud);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.05);
	seg.setMaxIterations(100);
	pcl::ModelCoefficients coeff;
	pcl::PointIndices index;
	seg.segment(index, coeff);
	//判断景物是否以平面为主
	bool planeFlag = false;
	cout << "plane rate: " << static_cast<float>(index.indices.size()) / objectPointCloud->size() << endl;
	if (static_cast<float>(index.indices.size()) / objectPointCloud->size() > th){
		planeFlag = true;
	}
		

	if (!planeFlag){
		seg.setInputCloud(scenePointCloud);
		seg.segment(index, coeff);
		cout << "plane rate: " << static_cast<float>(index.indices.size()) / scenePointCloud->size() << endl;
		if (static_cast<float>(index.indices.size()) / scenePointCloud->size() > th){
			planeFlag = true;
		}			
	}
	return planeFlag;
}