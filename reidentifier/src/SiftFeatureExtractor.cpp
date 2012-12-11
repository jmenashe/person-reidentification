#include "SiftFeatureExtractor.h"

SiftFeatureExtractor::SiftFeatureExtractor() : FeatureExtractor() {
  cv::initModule_nonfree();
  _detector = cv::FeatureDetector::create("Dense");
  _extractor = cv::DescriptorExtractor::create("SIFT");
}

PidMat SiftFeatureExtractor::extractFeature(const cv::Mat& image) {
  assert(image.channels() == 1);

  std::vector<cv::KeyPoint> keypoints;
  PidMat sifts;

  _detector->detect(image, keypoints);
  _extractor->compute(image, keypoints, sifts);
  if(!_isQuantized)
    sifts = sifts.reshape(1, sifts.rows * sifts.cols);
  if(_vocabBuilt)
    sifts = quantize(sifts);
  return sifts;
}
