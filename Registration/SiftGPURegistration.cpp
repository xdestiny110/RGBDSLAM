#include "SiftGPURegistration.h"
#include <GL/glew.h>

SiftGPUExtractor::SiftGPUExtractor()
{
	sift = new SiftGPU;
	char* parm[] = { "-fo", "-1", "-v", "0", "-e", "10.0", "-t", "0.004" };
	int parmNum = sizeof(parm) / sizeof(char*);
	sift->ParseParam(parmNum, parm);
	int siftResult;
	siftResult = static_cast<int>(sift->CreateContextGL());
	if (siftResult != SiftGPU::SIFTGPU_FULL_SUPPORTED){
		cerr << "ERROR: SIFTGPU cannot create contextGL!" << endl;
		exit(0);
	}
}

ExtractorBase& SiftGPUExtractor::Extract()
{	
	int siftResult = sift->RunSIFT(rgbImage.cols, rgbImage.rows, rgbImage.data, GL_RGB, GL_UNSIGNED_BYTE);
	if(siftResult == 0){
		cerr<<"ERROR: run SIFTGPU failed!"<<endl;
		exit(0);
	}

	int numOfFeature = sift->GetFeatureNum();
	keysOwn.clear();
	descriptorsOwn.clear();
	keysOwn.resize(numOfFeature);
	descriptorsOwn.resize(128 * numOfFeature);
	sift->GetFeatureVector(&keysOwn[0], &descriptorsOwn[0]);
	
	keys.resize(numOfFeature);
	descriptors = cv::Mat(numOfFeature, 128, CV_32F, cv::Scalar(0));
	for(int i = 0;i<numOfFeature;i++){
		keys[i].pt.x = keysOwn[i].x;
		keys[i].pt.y = keysOwn[i].y;
		keys[i].angle = keysOwn[i].o;
		keys[i].size = keysOwn[i].s;
		for (int j = 0; j < 128; j++)
			descriptors.at<float>(i, j) = descriptorsOwn[i * 128 + j];
	}
	
	return *this;
}

ExtractorBase& SiftGPUExtractor::ComputeDescriptors(){
	
	vector<SiftGPU::SiftKeypoint> k(keys.size());
	for (int i = 0; i < keys.size(); i++){
		k[i].x = keys[i].pt.x;
		k[i].y = keys[i].pt.y;
		k[i].o = keys[i].angle;
		k[i].s = keys[i].size;
	}
	sift->SetKeypointList(k.size(), &k[0]);
	int siftResult = sift->RunSIFT(rgbImage.cols, rgbImage.rows, rgbImage.data, GL_RGB, GL_UNSIGNED_BYTE);
	if (siftResult == 0){
		cerr << "ERROR: run SIFTGPU failed!" << endl;
		exit(0);
	}

	int numOfFeature = sift->GetFeatureNum();
	keysOwn.clear();
	descriptorsOwn.clear();
	keysOwn.resize(numOfFeature);
	descriptorsOwn.resize(128 * numOfFeature);
	sift->GetFeatureVector(&keysOwn[0], &descriptorsOwn[0]);

	keys.resize(numOfFeature);
	descriptors = cv::Mat(numOfFeature, 128, CV_32F, cv::Scalar(0));
	for (int i = 0; i<numOfFeature; i++){
		keys[i].pt.x = keysOwn[i].x;
		keys[i].pt.y = keysOwn[i].y;
		keys[i].angle = keysOwn[i].o;
		keys[i].size = keysOwn[i].s;
		for (int j = 0; j < 128; j++)
			descriptors.at<float>(i, j) = descriptorsOwn[i * 128 + j];
	}

	return *this;
}

SiftGPUMatcher::SiftGPUMatcher()
{
	matcher = new SiftMatchGPU;
	matcher->VerifyContextGL();
}

SiftGPUMatcher::SiftGPUMatcher(ExtractorBase& extractor1, ExtractorBase& extractor2)
{
	matcher = new SiftMatchGPU;
	matcher->VerifyContextGL();
	SetDescriptorsInObject(extractor1);
	SetDescriptorsInScene(extractor2);
}

MatcherBase& SiftGPUMatcher::SetDescriptorsInObject(ExtractorBase& extractorInObject)
{
	SiftGPUExtractor& _extractor = dynamic_cast<SiftGPUExtractor&>(extractorInObject);
	matcher->SetDescriptors(0, _extractor.GetNumOfFeature(), &_extractor.descriptorsOwn[0]);
	keypointInObject.clear();
	keypointInObject = _extractor.GetKeys();
	return *this;
}

MatcherBase& SiftGPUMatcher::SetDescriptorsInScene(ExtractorBase& extractorInObject)
{
	SiftGPUExtractor& _extractor = dynamic_cast<SiftGPUExtractor&>(extractorInObject);
	matcher->SetDescriptors(1, _extractor.GetNumOfFeature(), &_extractor.descriptorsOwn[0]);
	keypointInScene.clear();
	keypointInScene = _extractor.GetKeys();
	return *this;
}

MatcherBase& SiftGPUMatcher::Match()
{
	int(*match_buf)[2] = new int[keypointInObject.size()][2];
	numOfMatches = matcher->GetSiftMatch(keypointInObject.size(), match_buf, 0.7f, 0.65f);
	matchPairs.clear();
	
	for (int i = 0; i < numOfMatches; i++)
	{
		double sum = 0;
		matchPairs.push_back(cv::DMatch(match_buf[i][0], match_buf[i][1], 0.0));
	}
	delete[] match_buf;
	return *this;
}

RegistrationBase& SiftGPURegistration::SetObject(const cv::Mat& _rgbImage, const cv::Mat& _depthImage)
{
	rgbObjectImg = _rgbImage;
	depthObjectImg = _depthImage;
	if (objectExtractor.use_count() == 0)
		objectExtractor = make_shared<SiftGPUExtractor>();
	objectExtractor->SetRGBFile(rgbObjectImg);
	objectExtractor->SetKeys(vector<cv::KeyPoint>());
	objectExtractor->SetDescriptors(cv::Mat());
	return *this;
}

RegistrationBase& SiftGPURegistration::SetObject(const cv::Mat& _rgbImage, const cv::Mat& _depthImage,
		const vector<cv::KeyPoint>& _keys, const cv::Mat& _des)
{
	SetObject(_rgbImage, _depthImage);
	objectExtractor->SetKeys(_keys);
	objectExtractor->SetDescriptors(_des);
	return *this;
}

RegistrationBase& SiftGPURegistration::SetScene(const cv::Mat& _rgbImage, const cv::Mat& _depthImage)
{
	rgbSceneImg = _rgbImage;
	depthSceneImg = _depthImage;
	if (sceneExtractor.use_count() == 0)
		sceneExtractor = make_shared<SiftGPUExtractor>();
	sceneExtractor->SetRGBFile(rgbSceneImg);
	sceneExtractor->SetKeys(vector<cv::KeyPoint>());
	sceneExtractor->SetDescriptors(cv::Mat());
	return *this;
}

RegistrationBase& SiftGPURegistration::SetScene(const cv::Mat& _rgbImage, const cv::Mat& _depthImage,
	const vector<cv::KeyPoint>& _keys, const cv::Mat& _des)
{
	SetScene(_rgbImage, _depthImage);
	sceneExtractor->SetKeys(_keys);
	sceneExtractor->SetDescriptors(_des);
	return *this;
}

RegistrationBase& SiftGPURegistration::Apply()
{
	if (objectExtractor->GetKeys().empty())
		objectExtractor->Extract();
	if (objectExtractor->GetDescriptors().rows == 0)
		objectExtractor->ComputeDescriptors();
	if (sceneExtractor->GetKeys().empty())
		sceneExtractor->Extract();
	if (sceneExtractor->GetDescriptors().rows == 0)
		sceneExtractor->ComputeDescriptors();
	if (matcher.use_count()==0)
		matcher = make_shared<SiftGPUMatcher>(*objectExtractor, *sceneExtractor);
	else{
		matcher->SetDescriptorsInObject(*objectExtractor);
		matcher->SetDescriptorsInScene(*sceneExtractor);
	}

	matcher->Match();

	SiftGPUMatcher *matcherSym = new SiftGPUMatcher(*sceneExtractor, *objectExtractor);
	matcherSym->Match();

	vector<cv::DMatch> symMatches;
	for (auto it = matcher->GetMatchPairs().begin(); it != matcher->GetMatchPairs().end(); it++){
		for (auto jt = matcherSym->GetMatchPairs().begin(); jt != matcherSym->GetMatchPairs().end(); jt++){
			if (jt->trainIdx == it->queryIdx && jt->queryIdx == it->trainIdx)
				symMatches.push_back(*it);
		}
	}

	matcher->SetMatchPairs(symMatches);
	delete matcherSym;
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudSrc(new pcl::PointCloud<pcl::PointXYZ>),
		pointCloudTgt(new pcl::PointCloud<pcl::PointXYZ>);
	vector<cv::DMatch> matches = matcher->GetMatchPairs();
	vector<cv::KeyPoint> keypointSrc = matcher->GetKeypointInObject(),
		keypointTgt = matcher->GetKeypointInScene();

	for (auto it = matches.begin(); it != matches.end();)
	{
		uint16_t idensity = depthObjectImg.at<uint16_t>(static_cast<int>(keypointSrc[it->queryIdx].pt.y),
			static_cast<int>(keypointSrc[it->queryIdx].pt.x));
		pcl::PointXYZ ptSrc = Miscellaneous<pcl::PointXYZ>::Point2DTo3D(keypointSrc[it->queryIdx], idensity);
		idensity = depthSceneImg.at<uint16_t>(static_cast<int>(keypointTgt[it->trainIdx].pt.y),
			static_cast<int>(keypointTgt[it->trainIdx].pt.x));			
		pcl::PointXYZ ptTgt = Miscellaneous<pcl::PointXYZ>::Point2DTo3D(keypointTgt[it->trainIdx], idensity);
		if (ptSrc.z <0.0000001 || ptSrc.z>Miscellaneous<>::zMax || ptTgt.z<0.0000001 || ptTgt.z>Miscellaneous<>::zMax)
		{
			it = matches.erase(it);
			continue;
		}
		else
			it++;
		pointCloudSrc->push_back(ptSrc);
		pointCloudTgt->push_back(ptTgt);
	}
	Ransac<pcl::PointXYZ> ransac(pointCloudSrc, pointCloudTgt);
	transformationMatrix = ransac.GetTransforMat();
	vector<int> inlier = ransac.GetInliersIndex();
	matchesAfterRansac.clear();
	int minXObj = 10000, minYObj = 10000, maxXObj = -10000, maxYObj = -10000,
		minXSce = 10000, minYSce = 10000, maxXSce = -10000, maxYSce = -10000;
	for (auto it = inlier.begin(); it != inlier.end(); it++){
		matchesAfterRansac.push_back(*(matches.begin() + (*it)));
		cv::KeyPoint kptObj = matcher->GetKeypointInObject()[matchesAfterRansac.back().queryIdx],
			kptSce = matcher->GetKeypointInScene()[matchesAfterRansac.back().trainIdx];
		minXObj = (kptObj.pt.x < minXObj) ? kptObj.pt.x : minXObj;
		minYObj = (kptObj.pt.y < minYObj) ? kptObj.pt.y : minYObj;
		minXSce = (kptSce.pt.x < minXSce) ? kptSce.pt.x : minXSce;
		minYSce = (kptSce.pt.y < minYSce) ? kptSce.pt.y : minYSce;
		maxXObj = (kptObj.pt.x > maxXObj) ? kptObj.pt.x : maxXObj;
		maxYObj = (kptObj.pt.y > maxYObj) ? kptObj.pt.y : maxYObj;
		maxXSce = (kptSce.pt.x > maxXSce) ? kptSce.pt.x : maxXSce;
		maxYSce = (kptSce.pt.y > maxYSce) ? kptSce.pt.y : maxYSce;
	}

	robustFlag1 = ((float)maxXObj - minXObj)*(maxYObj - minYObj) / rgbObjectImg.rows / rgbObjectImg.cols;
	robustFlag2 = ((float)maxXSce - minXSce)*(maxYSce - minYSce) / rgbSceneImg.rows / rgbSceneImg.cols;

	return *this;
}

RegistrationBase& SiftGPURegistration::ApplyWith5Pt()
{
	if (matcher.use_count() == 0)
		Apply();
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudSrc(new pcl::PointCloud<pcl::PointXYZ>),
		pointCloudTgt(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXY>::Ptr pointCloudSrc2D(new pcl::PointCloud<pcl::PointXY>),
		pointCloudTgt2D(new pcl::PointCloud<pcl::PointXY>);
	vector<cv::DMatch> matches = matcher->GetMatchPairs();
	vector<cv::KeyPoint> keypointSrc = matcher->GetKeypointInObject(),
		keypointTgt = matcher->GetKeypointInScene();
	for (auto it = matches.begin(); it != matches.end();it++)
	{
		uint16_t idensity = depthObjectImg.at<uint16_t>(static_cast<int>(keypointSrc[it->queryIdx].pt.y),
			static_cast<int>(keypointSrc[it->queryIdx].pt.x));
		pcl::PointXYZ ptSrc = Miscellaneous<pcl::PointXYZ>::Point2DTo3D(keypointSrc[it->queryIdx], idensity);
		idensity = depthSceneImg.at<uint16_t>(static_cast<int>(keypointTgt[it->trainIdx].pt.y),
			static_cast<int>(keypointTgt[it->trainIdx].pt.x));
		pcl::PointXYZ ptTgt = Miscellaneous<pcl::PointXYZ>::Point2DTo3D(keypointTgt[it->trainIdx], idensity);
		pointCloudSrc->push_back(ptSrc);
		pointCloudTgt->push_back(ptTgt);
		pcl::PointXY srcXYPoint, tgtXYPoint;
		srcXYPoint.x = keypointSrc[it->queryIdx].pt.x;
		srcXYPoint.y = keypointSrc[it->queryIdx].pt.y;
		tgtXYPoint.x = keypointTgt[it->trainIdx].pt.x;
		tgtXYPoint.y = keypointTgt[it->trainIdx].pt.y;
		pointCloudSrc2D->push_back(srcXYPoint);
		pointCloudTgt2D->push_back(tgtXYPoint);
	}
	Ransac5pt ransac(pointCloudSrc, pointCloudTgt, pointCloudSrc2D, pointCloudTgt2D);
	transformationMatrix = ransac.GetTransforMat();
	vector<int> inlier = ransac.GetInliersIndex();
	inlierNum5Point = ransac.GetInliersNum();
	inlierNum5PointWithDepth = ransac.inlierNumWithDepth;
	matchesAfterRansac.clear();

	int minXObj = 10000, minYObj = 10000, maxXObj = -10000, maxYObj = -10000,
		minXSce = 10000, minYSce = 10000, maxXSce = -10000, maxYSce = -10000;
	for (auto it = inlier.begin(); it != inlier.end(); it++){
		matchesAfterRansac.push_back(*(matches.begin() + (*it)));
		cv::KeyPoint kptObj = matcher->GetKeypointInObject()[matchesAfterRansac.back().queryIdx],
			kptSce = matcher->GetKeypointInScene()[matchesAfterRansac.back().trainIdx];
		minXObj = (kptObj.pt.x < minXObj) ? kptObj.pt.x : minXObj;
		minYObj = (kptObj.pt.y < minYObj) ? kptObj.pt.y : minYObj;
		minXSce = (kptSce.pt.x < minXSce) ? kptSce.pt.x : minXSce;
		minYSce = (kptSce.pt.y < minYSce) ? kptSce.pt.y : minYSce;
		maxXObj = (kptObj.pt.x > maxXObj) ? kptObj.pt.x : maxXObj;
		maxYObj = (kptObj.pt.y > maxYObj) ? kptObj.pt.y : maxYObj;
		maxXSce = (kptSce.pt.x > maxXSce) ? kptSce.pt.x : maxXSce;
		maxYSce = (kptSce.pt.y > maxYSce) ? kptSce.pt.y : maxYSce;
	}
		
	robustFlag1 = ((float)maxXObj - minXObj)*(maxYObj - minYObj) / rgbObjectImg.rows / rgbObjectImg.cols;
	robustFlag2 = ((float)maxXSce - minXSce)*(maxYSce - minYSce) / rgbSceneImg.rows / rgbSceneImg.cols;

	return *this;
}