#include "Location.h"
#include "Miscellaneous.h"
#include <pcl/io/ply_io.h>
#include <memory>
using namespace std;

int main(int argc,char** argv)
{
	string bathPath = argv[1], keyFramePath = argv[2];
	string rgbIndexFile = bathPath + "rgb.txt",
		depthIndexFile = bathPath + "depth.txt",
		rgbFile, depthFile;
	ifstream depthIndexStream(depthIndexFile, ios::in),
		rgbIndexStream(rgbIndexFile, ios::in);

	shared_ptr<Location> l = make_shared<Location>(keyFramePath);
	int ind = -1;
	while (!depthIndexStream.eof() && !rgbIndexStream.eof()){
		rgbIndexStream >> rgbFile >> rgbFile;
		depthIndexStream >> depthFile >> depthFile;
		ind++;
		//if (ind < 43) continue;

		assert(!rgbFile.empty() && !depthFile.empty());
		rgbFile = bathPath + rgbFile;
		depthFile = bathPath + depthFile;

		cv::Mat objectRGBImage = cv::imread(rgbFile),
			objectDepthImage = cv::imread(depthFile, cv::IMREAD_ANYDEPTH);
		l->SetObject(objectRGBImage, objectDepthImage);
		Eigen::MatrixXf transMat = l->FindKeyFrame(ind);
		if (transMat != Eigen::Matrix4f::Identity()){
			std::cout << transMat << endl;
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr objectPointCloud
				= Miscellaneous<>::GeneratePointCloud(objectDepthImage, objectRGBImage, 0.01),
				transPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			pcl::transformPointCloud(*objectPointCloud, *transPointCloud, transMat);
			pcl::PLYWriter writer;
			writer.write(string(argv[1]) + "Location.ply", *transPointCloud, true);
			return 1;
		}
	}

	std::cout << "Cannot location!" << endl;
	return 0;
}