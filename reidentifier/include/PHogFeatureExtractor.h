#ifndef PHOG_EXTRACTOR_H
#define PHOG_EXTRACTOR_H

#include "FeatureExtractor.h"
#include <opencv2/highgui/highgui.hpp>

class PHogFeatureExtractor : public FeatureExtractor {
  public:
    PHogFeatureExtractor();
    PidMat extractFeature(const cv::Mat&);
};

#endif
