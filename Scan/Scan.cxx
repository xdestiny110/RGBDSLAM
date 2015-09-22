#include "common.h"
#include "Miscellaneous.h"
#include "GraphEnd.h"
#include "SiftGPURegistration.h"

void method(GraphEnd& g){
	int status2 = g.CheckClouse();
	cout << "status2: " << status2 << endl;
	switch (status2)
	{
	case 1:
		g.preTrust = true;
		break;
	case 2:
		g.preTrust = true;
		g.GenerateKeyFrame(KeyFrame(g.keyframeVector.back().robotPos*g.reg->GetTransformationMatrix(),
			g.reg->GetRGBImageInObject(), g.reg->GetDepthImageInObject(), g.reg->GetKeysInObject(),
			g.keyframeVector.size()));
		g.GenerateEdge(g.keyframeVector.rbegin()->imgId, (g.keyframeVector.rbegin() + 1)->imgId,
			g.reg->GetTransformationMatrix(), 25);
		g.FindClouse();
		break;
	case 3:
		g.preTrust = false;
		break;
	case 4:
		g.preTrust = false;
		g.GenerateKeyFrame(KeyFrame(g.keyframeVector.back().robotPos*g.reg->GetTransformationMatrix(),
			g.reg->GetRGBImageInObject(), g.reg->GetDepthImageInObject(), g.reg->GetKeysInObject(),
			g.keyframeVector.size()));
		g.GenerateEdge(g.keyframeVector.rbegin()->imgId, (g.keyframeVector.rbegin() + 1)->imgId,
			g.reg->GetTransformationMatrix(), 25);
		g.FindClouse();
		break;
	case 5:
		cout << "wait for g2o" << endl;
		g.preTrust = false;
		g.GenerateKeyFrame(KeyFrame(g.keyframeVector.back().robotPos*g.reg->GetTransformationMatrix(),
			g.reg->GetRGBImageInObject(), g.reg->GetDepthImageInObject(), g.reg->GetKeysInObject(),
			g.keyframeVector.size()));
		g.FindClouse();
		break;
	default:cerr << "status error!" << endl; cv::waitKey(); break;
	}
}

int main(int argc, char** argv)
{
	string bathPath = argv[1];
	string rgbIndexFile = bathPath + "rgb.txt",
		depthIndexFile = bathPath + "depth.txt";
	ifstream depthIndexStream(depthIndexFile, ios::in),
		rgbIndexStream(rgbIndexFile, ios::in);

	GraphEnd *g = new GraphEnd;
	string rgbFile, depthFile;
	rgbIndexStream >> rgbFile >> rgbFile;
	depthIndexStream >> depthFile >> depthFile;
	rgbFile = bathPath + rgbFile;
	depthFile = bathPath + depthFile;

	g2o::Vector7d initPos;
	initPos << 0, 0, 0, 0, 0, 0, 1;//DS4Data  data2015-5-15 indoor data2015-5-27 data2015-6-19
	//initPos << -2.2583, -2.3803, 0.5900, 0.1906, 0.7165, -0.6513, -0.1613;//rgbd_dataset_freiburg2_pioneer_slam2
	//initPos << 1.2956, -1.6787, 0.5734, -0.6193, 0.3968, -0.3608, 0.5734;//rgbd_dataset_freiburg2_pioneer_slam2_del
	//initPos << 1.4908, -1.1707, 0.6603, 0.8959, 0.0695, -0.0461, -0.4364;//rgbd_dataset_freiburg1_floor
	//initPos << 0.1163, -1.1498, 1.4015, -0.5721, 0.6521, -0.3565, 0.3469;//rgbd_dataset_freiburg2_xyz
	//initPos << 1.5015, 0.9306, 1.4503, 0.8703, 0.1493, -0.1216, -0.4533;//rgbd_dataset_freiburg1_desk2
	//initPos << -2.3173, -2.1669, 0.5874, -0.7161, 0.2093, -0.1837, 0.6400;//rgbd_dataset_freiburg2_pioneer_slam
	//initPos << 2.1684, -0.3086, 0.5643, -0.7365, -0.0948, 0.0941, 0.6631;//rgbd_dataset_freiburg2_pioneer_slam_delete
	//initPos << -1.8476, 2.9446, 0.5268, -0.6810, -0.2887, 0.2888, 0.6078;//rgbd_dataset_freiburg2_pioneer_slam3
	Eigen::Isometry3d initPosI3d = g2o::internal::fromVectorQT(initPos);
	Eigen::MatrixXf initPosMat(4, 4);
	initPosMat = Miscellaneous<>::Isometry3dToMatrixXf(initPosI3d);
	SiftGPUExtractor e;
	//SiftFeatExtractor e;
	g->currentRGBImage = cv::imread(rgbFile);
	g->currentDepthImage = cv::imread(depthFile, CV_LOAD_IMAGE_ANYDEPTH);
	e.SetRGBFile(g->currentRGBImage);
	e.Extract();
	g->GenerateKeyFrame(
		KeyFrame(initPosMat, e.GetRGBImage(), g->currentDepthImage, e.GetKeys(), 0), true);
	g->preRGBImage = g->currentRGBImage;
	g->preDepthImage = g->currentDepthImage;
	int keyframeId = 1, count = 0;
	int imageStep = 1;
	while (!rgbIndexStream.eof() && !depthIndexStream.eof())
	{
		rgbIndexStream >> rgbFile >> rgbFile;
		depthIndexStream >> depthFile >> depthFile;
		rgbFile = bathPath + rgbFile;
		depthFile = bathPath + depthFile;
		count++;
		if (count > 1300) break;
		if (count % imageStep)
			continue;
		cout << "##################" << endl;
		cout << "image id: " << count << endl;
		g->currentRGBImage = cv::imread(rgbFile);
		g->currentDepthImage = cv::imread(depthFile, CV_LOAD_IMAGE_ANYDEPTH);
		int status = g->CheckClouse();
		cout << "status: " << status << endl;
		switch (status)
		{
		case 1:			
			g->preTrust = true;
			break;
		case 2:
			g->preTrust = true;
			g->GenerateKeyFrame(KeyFrame(g->keyframeVector.back().robotPos*g->reg->GetTransformationMatrix(),
				g->reg->GetRGBImageInObject(), g->reg->GetDepthImageInObject(), g->reg->GetKeysInObject(),
				g->keyframeVector.size()));
			g->GenerateEdge(g->keyframeVector.rbegin()->imgId, (g->keyframeVector.rbegin() + 1)->imgId,
				g->reg->GetTransformationMatrix(), 25);
			g->FindClouse();
			break;
		case 3:
			//若上一帧可信且不为关键帧，则将上一帧作为关键帧再重算
			if (g->preTrust && g->keyframeVector.back().rgbImage.data != g->preRGBImage.data){
				cout << "use pre-frame" << endl;
				g->GenerateKeyFrame(KeyFrame(g->keyframeVector.back().robotPos*g->preTransformation,
					g->preRGBImage, g->preDepthImage, g->preKeyPoint, g->keyframeVector.size()));
				g->GenerateEdge(g->keyframeVector.rbegin()->imgId, (g->keyframeVector.rbegin() + 1)->imgId,
					g->preTransformation, 25);
				method(*g);				
			}
			//否则保留ICP结果
			else
				g->preTrust = false;
			break;
		case 4:
			//若上一帧可信且不为关键帧，则将上一帧作为关键帧再重算
			if (g->preTrust && g->keyframeVector.back().rgbImage.data != g->preRGBImage.data){
				cout << "use pre-frame" << endl;
				g->GenerateKeyFrame(KeyFrame(g->keyframeVector.back().robotPos*g->preTransformation,
					g->preRGBImage, g->preDepthImage, g->preKeyPoint, g->keyframeVector.size()));
				g->GenerateEdge(g->keyframeVector.rbegin()->imgId, (g->keyframeVector.rbegin() + 1)->imgId,
					g->preTransformation, 25);
				method(*g);
			}
			//否则以ICP结果建立关键帧
			else{
				g->preTrust = false;
				g->GenerateKeyFrame(KeyFrame(g->keyframeVector.back().robotPos*g->reg->GetTransformationMatrix(),
					g->reg->GetRGBImageInObject(), g->reg->GetDepthImageInObject(), g->reg->GetKeysInObject(),
					g->keyframeVector.size()));
				g->GenerateEdge(g->keyframeVector.rbegin()->imgId, (g->keyframeVector.rbegin() + 1)->imgId,
					g->reg->GetTransformationMatrix(), 25);
				g->FindClouse();
			}				
			break;
		case 5:
			//转移矩阵计算错误
			//无论上一帧如何都需要建立关键帧
			if (g->keyframeVector.back().rgbImage.data != g->preRGBImage.data){
				g->GenerateKeyFrame(KeyFrame(g->keyframeVector.back().robotPos*g->preTransformation,
					g->preRGBImage, g->preDepthImage, g->preKeyPoint, g->keyframeVector.size()));
				g->GenerateEdge(g->keyframeVector.rbegin()->imgId, (g->keyframeVector.rbegin() + 1)->imgId,
					g->preTransformation, 25);
				method(*g);
			}
			else{
				cout << "wait for g2o" << endl;
				g->preTrust = false;
				g->GenerateKeyFrame(KeyFrame(g->keyframeVector.back().robotPos*g->reg->GetTransformationMatrix(),
					g->reg->GetRGBImageInObject(), g->reg->GetDepthImageInObject(), g->reg->GetKeysInObject(),
					g->keyframeVector.size()));
				g->FindClouse();
			}
			break;
			//若无法建立则期待g2o求解
		default:cerr << "status error!" << endl; cv::waitKey(); break;
		}
		g->preRGBImage = g->currentRGBImage;
		g->preDepthImage = g->currentDepthImage;
		g->preTransformation = g->reg->GetTransformationMatrix();
		g->preKeyPoint = g->reg->GetKeysInObject();
	}

	g->SaveGraph(bathPath + "KeyFrame/BeforeOptimize.g2o");
	g->SaveKeyFrame(bathPath + "KeyFrame/");
	g->Optimize();
	g->SaveGraph(bathPath + "KeyFrame/AfterOptimize.g2o");
	g->SaveTrajectory(bathPath + "KeyFrame/Trajectory.txt");

	delete g;
	depthIndexStream.close();
	rgbIndexStream.close();

	return 0;
}