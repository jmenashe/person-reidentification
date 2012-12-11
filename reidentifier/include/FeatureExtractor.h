#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/flann/flann.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include "Typedefs.h"

typedef std::vector<std::vector<PidMat> > FeatureSet;

class FeatureExtractor {
  public:
    FeatureExtractor();
    ~FeatureExtractor();
    PidMat processImage(const cv::Mat&);
    virtual PidMat extractFeature(const cv::Mat&) = 0;
    std::vector<PidMat> quantize(std::vector<PidMat>&);
    PidMat quantize(PidMat&);
    static std::vector<PidMat> combine(FeatureSet&);

    void reset();
    std::vector<PidMat> getFeatures();
    void clearFeatures();

    void fromYaml(const YAML::Node& node);
    void toYaml(YAML::Emitter&) const;

    void setLabels(const std::vector<int>&);
    void setQuantized(bool);
    void setQuantizePoint(int);

    bool isQuantized() const;
    bool vocabularyBuilt() const;

  protected:
    int _vocabularySize, _vocabularySamples;
    bool _isQuantized;
    bool _vocabBuilt;
  
  private:
    void buildVocabulary();
    
    cv::flann::Index* _index;
    PidMat _vocabulary;
    std::vector<PidMat> _unquantized, _quantized;

    int _quantizePoint;
    std::vector<int> _labels;
};

YAML::Emitter& operator<< (YAML::Emitter&, const FeatureExtractor&);
void operator>> (const YAML::Node&, FeatureExtractor&);

#endif
