#include "ransac5pt.h"
#include "Miscellaneous.h"

void Ransac5pt::ransacMain(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud1,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud2,
	pcl::PointCloud<pcl::PointXY>::Ptr imgPtCld1,
	pcl::PointCloud<pcl::PointXY>::Ptr imgPtCld2)
{

	transforMat = Eigen::Matrix4f::Identity();
	int num_pts = pointCloud1->points.size();

	v2_t *k1_pts = new v2_t[num_pts];
	v2_t *k2_pts = new v2_t[num_pts];
	v3_t *k1_pts_3d = new v3_t[num_pts];
	v3_t *k2_pts_3d = new v3_t[num_pts];

	for (int i = 0; i < num_pts; i++) {
		k1_pts[i] = v2_new(imgPtCld1->points[i].x, (479 - imgPtCld1->points[i].y));
		k2_pts[i] = v2_new(imgPtCld2->points[i].x, (479 - imgPtCld2->points[i].y));
		k1_pts_3d[i] = v3_new(pointCloud1->points[i].x, (-pointCloud1->points[i].y), (-pointCloud1->points[i].z));
		k2_pts_3d[i] = v3_new(pointCloud2->points[i].x, (-pointCloud2->points[i].y), (-pointCloud2->points[i].z));
	}

	double K[9];
	double focalLength = 620.0, centerX = 319.5, centerY = 239.5;
	K[0] = focalLength;  K[1] = 0.0;		  K[2] = centerX;
	K[3] = 0.0;			 K[4] = focalLength;  K[5] = centerY;
	K[6] = 0.0;			 K[7] = 0.0;		  K[8] = 1.0;

	double R[9], t[3];
	int *inlierList = new int[num_pts];
	for (int i = 0; i < num_pts; i++) inlierList[i] = 0;
	inliersIndex.clear();
	inlierNumWithDepth = 0;
	inliersNum = 0;
	if (num_pts < 5){
		cerr << "no enough points for ransac!" << endl;
		transforMat = Eigen::Matrix4f::Identity();
		return;
	}
	inliersNum = compute_pose_ransac(num_pts, k1_pts, k2_pts, k1_pts_3d, k2_pts_3d,
		K, K, 2, maxIterNum, R, t, inlierList);

	for (int i = 0; i < num_pts; i++)
	if (inlierList[i] > 0) {
		double z1 = pointCloud1->points[i].z, z2 = pointCloud2->points[i].z;
		inliersIndex.push_back(i);
		if (z1 > Miscellaneous<>::zMax || z1 < 0.01 || z2 > Miscellaneous<>::zMax || z2 < 0.01) continue;
		inlierNumWithDepth++;
	}
	cout << "5 point match number: " << inlierNumWithDepth << '/' << inliersNum << endl;

	delete[] k1_pts;
	delete[] k2_pts;
	delete[] k1_pts_3d;
	delete[] k2_pts_3d;
	delete[] inlierList;

	transforMat(0, 0) = R[0]; transforMat(0, 1) = R[1]; transforMat(0, 2) = R[2]; transforMat(0, 3) = t[0];
	transforMat(1, 0) = R[3]; transforMat(1, 1) = R[4]; transforMat(1, 2) = R[5]; transforMat(1, 3) = t[1];
	transforMat(2, 0) = R[6]; transforMat(2, 1) = R[7]; transforMat(2, 2) = R[8]; transforMat(2, 3) = t[2];
	transforMat(3, 0) = 0.0; transforMat(3, 1) = 0.0; transforMat(3, 2) = 0.0; transforMat(3, 3) = 1.0;

	Eigen::Matrix4f rotX;
	rotX << 1, 0, 0, 0,
		0, -1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1;
	transforMat = rotX*transforMat*rotX;
	//Eigen::Matrix3f tmp = transforMat.block<3, 3>(0, 0);
	//transforMat.block<3, 3>(0, 0) = tmp.transpose();
	//transforMat(0, 3) = -transforMat(0, 3);
	//transforMat(1, 3) = -transforMat(1, 3);
	//transforMat(2, 3) = -transforMat(2, 3);
}