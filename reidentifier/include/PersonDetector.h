#ifndef PERSON_DETECTOR_H
#define PERSON_DETECTOR_H

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <boost/foreach.hpp>

class PersonDetector {
    static const float OverlapThresh;
    static const float DetectThresh;
    static const int NumThreads = 8;
    std::string _modelDirectory;
    cv::LatentSvmDetector _detector;
  public:
    PersonDetector();
    bool load(std::string);
    void detect(const cv::Mat&,std::vector<cv::Rect>&);
};

#endif
