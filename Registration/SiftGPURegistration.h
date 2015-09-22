#ifndef SIFTGPUREGISTRATION_H_
#define SFITGPUREGISTRATION_H_

#include "RegistrationBase.h"
#include <SiftGPU.h>

class SiftGPUExtractor :public ExtractorBase
{
public:
	SiftGPUExtractor();
	ExtractorBase& Extract() override;
	ExtractorBase& ComputeDescriptors() override;
	ExtractorBase& SetKeys(const vector<cv::KeyPoint>& _keys) override
	{
		ExtractorBase::SetKeys(_keys);
		keysOwn.clear();
		keysOwn.resize(_keys.size());
		for (size_t i = 0; i < _keys.size();i++){
			keysOwn[i].x = _keys[i].pt.x;
			keysOwn[i].y = _keys[i].pt.y;
			keysOwn[i].s = _keys[i].size;
			keysOwn[i].o = _keys[i].angle;
		}
		return *this;
	}
	ExtractorBase& SetDescriptors(const cv::Mat& _des)
	{
		ExtractorBase::SetDescriptors(_des);
		descriptorsOwn.clear();
		descriptorsOwn.resize(_des.rows*_des.cols);
		for (int i = 0; i < _des.rows; i++){
			for (int j = 0; j < _des.cols; j++){
				descriptorsOwn[i * 128 + j] = _des.at<float>(i, j);
			}
		}
		return *this;
	}

	~SiftGPUExtractor()
	{
		delete sift;
	}
	vector<float> descriptorsOwn;
	vector<SiftGPU::SiftKeypoint> keysOwn;
private:
	SiftGPU* sift;
};

class SiftGPUMatcher :public MatcherBase
{
public:
	SiftGPUMatcher();
	SiftGPUMatcher(ExtractorBase& extractor1, ExtractorBase& extractor2);
	MatcherBase& SetDescriptorsInObject(ExtractorBase& extractorInObject) override;
	MatcherBase& SetDescriptorsInScene(ExtractorBase& extractorInObject) override;

	MatcherBase& Match() override;
	~SiftGPUMatcher()
	{
		delete matcher;
	}
private:
	SiftMatchGPU* matcher;
};

class SiftGPURegistration :public RegistrationBase
{
public:
	SiftGPURegistration(const cv::Mat& _rgbObject, const cv::Mat& _rgbScene,
		const cv::Mat& _depthObject, const cv::Mat& _depthScene)
	{		
		matcher = NULL;
		SetObject(_rgbObject, _depthObject);
		SetScene(_rgbScene, _depthScene);
	}
	RegistrationBase& SetObject(const cv::Mat& _rgbImage, const cv::Mat& _depthImage) override;
	RegistrationBase& SetObject(const cv::Mat& _rgbImage, const cv::Mat& _depthImage,
		const vector<cv::KeyPoint>& _keys, const cv::Mat& _des) override;
	RegistrationBase& SetScene(const cv::Mat& _rgbImage, const cv::Mat& _depthImage) override;
	RegistrationBase& SetScene(const cv::Mat& _rgbImage, const cv::Mat& _depthImage,
		const vector<cv::KeyPoint>& _keys,const cv::Mat& _des) override;
	RegistrationBase& Apply() override;	
	RegistrationBase& ApplyWith5Pt() override;
	~SiftGPURegistration()
	{

	}
private:
};
#endif