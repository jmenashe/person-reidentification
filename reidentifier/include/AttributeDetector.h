#ifndef ATTRIBUTE_DETECTOR_H
#define ATTRIBUTE_DETECTOR_H

#include "OcvSvm.h"
#include "LibsvmSvm.h"
#include "DlibSvm.h"
#include "Util.h"
#include "Typedefs.h"
#include "Parts.h"
#include "SiftFeatureExtractor.h"
#include "HogFeatureExtractor.h"
#include "PHogFeatureExtractor.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>

typedef std::vector<std::vector<PidMat> > FeatureSet;

class AttributeDetector {
  public:
    AttributeDetector(std::string, float, Parts::Location = Parts::WholeBody);
    ~AttributeDetector();
    bool hasAttribute(cv::Mat&, cv::Rect);
    bool hasAttribute(cv::Mat&, cv::Rect, float&);
    float getWeight();
    Parts::Location getDetectLoc();
    std::string getAttribute();
    void load();
    void save();
    void train(const std::vector<std::string>&, const std::vector<int>&);
    void train(const std::vector<std::string>&, const std::vector<cv::Rect>&, const std::vector<int>&);
  private:
    FeatureExtractor *_hogExtractor, *_siftExtractor;
    Svm* _svm;
    std::string _attribute;
    Parts::Location _attributeLoc;
    float _weight;
    float _threshold;
    bool _bagOfWords, _useHOG, _useSIFT;

    void trainAttribute(const std::vector<PidMat>&,const std::vector<int>&);

    void normalize(PidMat&);
    void normalize(std::vector<PidMat>&);
};

#endif
