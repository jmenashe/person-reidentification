#include "PersonDetector.h"

const float PersonDetector::OverlapThresh = 0.2f;
const float PersonDetector::DetectThresh = 0.5;

PersonDetector::PersonDetector() {
}

bool PersonDetector::load(std::string modelDirectory) {
  _modelDirectory = modelDirectory;
    std::vector<std::string> Models;
    Models.push_back(_modelDirectory + "/person.xml");
  _detector.load(Models);
  return(_detector.getClassCount());
}

void PersonDetector::detect(const cv::Mat& image, std::vector<cv::Rect>& DLoc) {
    std::vector<cv::LatentSvmDetector::ObjectDetection> detections;
    _detector.detect(image,detections,OverlapThresh,NumThreads);
    BOOST_FOREACH(cv::LatentSvmDetector::ObjectDetection TempDetect, detections) {
        if(TempDetect.score>DetectThresh)
            DLoc.push_back(TempDetect.rect);
    }
}

