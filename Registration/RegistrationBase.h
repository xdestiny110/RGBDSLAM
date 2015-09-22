#ifndef REGISTRATIONBASE_H_
#define REGISTRATIONBASE_H_

#include "common.h"
#include "Miscellaneous.h"
#include "Ransac.h"
#include <../5Point/ransac5pt.h>

class ExtractorBase
{
public:
	typedef shared_ptr<ExtractorBase> Ptr;
	ExtractorBase()
	{

	}
	virtual ExtractorBase& SetRGBFile(const cv::Mat _rgbImage)
	{
		//_rgbImage.copyTo(rgbImage);
		rgbImage = _rgbImage;
		return *this;
	}
	virtual const cv::Mat& GetRGBImage() const
	{
		return rgbImage;
	}
	virtual ExtractorBase& SetKeys(const vector<cv::KeyPoint>& _keys)
	{
		keys = _keys;
		return *this;
	}
	virtual const vector<cv::KeyPoint>& GetKeys() const
	{
		return keys;
	}
	virtual ExtractorBase& SetDescriptors(const cv::Mat& _des)
	{
		//_des.copyTo(descriptors);
		descriptors = _des;
		return *this;
	}
	virtual const cv::Mat& GetDescriptors() const
	{
		return descriptors;
	}
	virtual const int GetNumOfFeature() const
	{
		return keys.size();
	}
	virtual ExtractorBase& Extract() = 0;
	virtual ExtractorBase& ComputeDescriptors() = 0;
	virtual ~ExtractorBase()
	{
		
	}
protected:
	cv::Mat rgbImage;
	cv::Mat descriptors;
	vector<cv::KeyPoint> keys;
	
};

class MatcherBase
{
public:
	typedef shared_ptr<MatcherBase> Ptr;
	virtual MatcherBase& Match() = 0;
	virtual MatcherBase& SetDescriptorsInObject(ExtractorBase& extractorInObject) = 0;	
	virtual MatcherBase& SetDescriptorsInScene(ExtractorBase& extractorInScene) = 0;
	virtual const int GetNumOfMatches() const
	{
		return numOfMatches;
	}
	virtual const vector<cv::DMatch>& GetMatchPairs() const
	{
		return matchPairs;
	}
	virtual MatcherBase& SetMatchPairs(const vector<cv::DMatch> otherMatchPairs)
	{
		matchPairs = otherMatchPairs;
		return *this;
	}
	virtual const vector<cv::KeyPoint>& GetKeypointInObject() const
	{
		return keypointInObject;
	}
	virtual const vector<cv::KeyPoint>& GetKeypointInScene() const
	{
		return keypointInScene;
	}
	virtual ~MatcherBase()
	{

	}
protected:
	int numOfMatches;
	vector<cv::KeyPoint> keypointInObject, keypointInScene;
	vector<cv::DMatch> matchPairs;
};

class RegistrationBase
{
public:
	typedef shared_ptr<RegistrationBase> Ptr;
	RegistrationBase()
		:objectExtractor(NULL), sceneExtractor(NULL), matcher(NULL), robustFlag1(0.f), robustFlag2(0.f)
	{

	}
	virtual RegistrationBase& SetObject(const cv::Mat& _rgbImage, const cv::Mat& _depthImage) = 0;
	virtual RegistrationBase& SetObject(const cv::Mat& _rgbImage, const cv::Mat& _depthImage,
		const vector<cv::KeyPoint>& _keys, const cv::Mat& _des) = 0;
	virtual RegistrationBase& SetScene(const cv::Mat& _rgbImage, const cv::Mat& _depthImage) = 0;
	virtual RegistrationBase& SetScene(const cv::Mat& _rgbImage, const cv::Mat& _depthImage,
		const vector<cv::KeyPoint>& _keys,const cv::Mat& _des) = 0;
	virtual RegistrationBase& SetTransformationMatrix(Eigen::MatrixXf t)
	{
		transformationMatrix = t;
		return *this;
	}

	virtual const vector<cv::KeyPoint>& GetKeysInObject() const
	{
		return objectExtractor->GetKeys();
	}
	virtual const cv::Mat& GetDescriptorsInObject() const
	{
		return objectExtractor->GetDescriptors();
	}
	virtual const vector<cv::KeyPoint>& GetKeysInScene() const
	{
		return sceneExtractor->GetKeys();
	}
	virtual const cv::Mat& GetDescriptorsInScene() const
	{
		return sceneExtractor->GetDescriptors();
	}
	virtual const cv::Mat& GetRGBImageInObject() const
	{
		return rgbObjectImg;
	}
	virtual const cv::Mat& GetDepthImageInObject()  const
	{
		return depthObjectImg;
	}
	virtual const cv::Mat& GetRGBImageInScene() const
	{
		return rgbSceneImg;
	}
	virtual const cv::Mat& GetDepthImageInScene() const
	{
		return depthSceneImg;
	}
	virtual bool GetRobustFlag(float th = 0.2) const
	{
		return (robustFlag1 > th && robustFlag2 > th) ? true : false;
	}
	virtual RegistrationBase& Apply() = 0;
	virtual RegistrationBase& ApplyWith5Pt() = 0;
	virtual RegistrationBase& DrawMatch(cv::Mat& outputImg, bool verbose = true)
	{
		if (verbose)
			cv::drawMatches(rgbObjectImg, matcher->GetKeypointInObject(), rgbSceneImg, matcher->GetKeypointInScene(),
			matchesAfterRansac, outputImg);
		else{
			vector<cv::DMatch> temp(matchesAfterRansac.size());
			vector<cv::KeyPoint> kptObject(matchesAfterRansac.size()), kptScene(matchesAfterRansac.size());
			for (int i = 0; i < matchesAfterRansac.size();i++){
				temp[i].queryIdx = i;
				temp[i].trainIdx = i;
				kptObject[i] = matcher->GetKeypointInObject()[matchesAfterRansac[i].queryIdx];
				kptScene[i] = matcher->GetKeypointInScene()[matchesAfterRansac[i].trainIdx];
			}
			cv::drawMatches(rgbObjectImg, kptObject, rgbSceneImg, kptScene,
				temp, outputImg);
		}
		return *this;
	}

	virtual RegistrationBase& DrawMatchBeforeRansac(cv::Mat& outputImg)
	{
		cv::drawMatches(rgbObjectImg, matcher->GetKeypointInObject(), rgbSceneImg, matcher->GetKeypointInScene(),
			matcher->GetMatchPairs(), outputImg);
		return *this;
	}

	virtual const Eigen::MatrixXf& GetTransformationMatrix() const
	{
		return transformationMatrix;
	}

	virtual const int GetInliersNum() const
	{
		return matchesAfterRansac.size();
	}

	virtual const int GetMatchNumBeforeRansac() const
	{
		return matcher->GetMatchPairs().size();
	}

	virtual const int Get5PointInlierWithDepthNum() const
	{
		return inlierNum5PointWithDepth;
	}

	virtual const int Get5PointInlierNum() const
	{
		return inlierNum5Point;
	}

	virtual ~RegistrationBase()
	{
		
	}
protected:
	cv::Mat rgbObjectImg, rgbSceneImg, depthObjectImg, depthSceneImg;
	Eigen::MatrixXf transformationMatrix;
	vector<cv::DMatch> matchesAfterRansac;
	ExtractorBase::Ptr objectExtractor, sceneExtractor;
	MatcherBase::Ptr matcher;
	int inlierNum5Point, inlierNum5PointWithDepth;
	float robustFlag1, robustFlag2;
};
#endif