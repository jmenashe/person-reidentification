#ifndef HOG_EXTRACTOR_H
#define HOG_EXTRACTOR_H

#include "FeatureExtractor.h"

class HogFeatureExtractor : public FeatureExtractor {
  public:
    HogFeatureExtractor();
    PidMat extractFeature(const cv::Mat&);
  private:
    cv::Ptr<cv::HOGDescriptor> _extractor;

};

#endif
