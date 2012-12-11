#ifndef SIFT_FEATURE_EXTRACTOR_H
#define SIFT_FEATURE_EXRACTOR_H

#include "FeatureExtractor.h"

class SiftFeatureExtractor : public FeatureExtractor {
  public:
    SiftFeatureExtractor();
    PidMat extractFeature(const cv::Mat&);
  private:
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _extractor;
};

#endif
