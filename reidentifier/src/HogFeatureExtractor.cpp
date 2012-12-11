#include "HogFeatureExtractor.h"

HogFeatureExtractor::HogFeatureExtractor() {
  _extractor = new cv::HOGDescriptor();
  _vocabularySize = 25;
  _vocabularySamples = _vocabularySize * 50 / _extractor->nbins;
}

PidMat HogFeatureExtractor::extractFeature(const cv::Mat& image) {
  std::vector<float> hogs;
  std::vector<cv::Point> HogPoints;
  assert(image.channels() == 1);

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(64, 128));

  _extractor->compute(resized, hogs, cv::Size(8,8), cv::Size(0,0), HogPoints);
  PidMat feature(hogs.size(), 1);
  for(int i=0; i < hogs.size(); i++)
    feature(i,0) = hogs[i];
  if(_isQuantized) 
    feature = feature.reshape(1, _extractor->nbins);
  if(_vocabBuilt)
    feature = quantize(feature);
  return feature;
}


