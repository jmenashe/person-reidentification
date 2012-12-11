#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor() {
  _vocabBuilt = false;
  _index = 0;
  _vocabularySamples = 10;
  _vocabularySize = 100;
}

FeatureExtractor::~FeatureExtractor() {
  if(_index) delete _index;
}

PidMat FeatureExtractor::processImage(const cv::Mat& image) {
  PidMat feature = extractFeature(image);
  if(_vocabBuilt) {
    _quantized.push_back(feature);
  }
  else if (!_isQuantized || _unquantized.size() < _quantizePoint) {
    _unquantized.push_back(feature);
  }
  else {
    buildVocabulary();
    _quantized = quantize(_unquantized);
    _unquantized.clear();
    feature = quantize(feature);
    _quantized.push_back(feature);
  }
  return feature;
}

std::vector<PidMat> FeatureExtractor::getFeatures() {
  if(_isQuantized)
    return _quantized;
  return _unquantized;
}

void FeatureExtractor::clearFeatures() {
  _quantized.clear();
  _unquantized.clear();
}

std::vector<PidMat> FeatureExtractor::combine(FeatureSet& fset) {
  assert(fset.size() > 0);
  std::vector<PidMat> combined = fset[0];
  for(int i = 1; i < fset.size(); i++) {
    for(int j = 0; j < combined.size(); j++) {
      combined[j].push_back(fset[i][j]);
    }
  }
  return combined;
}

void FeatureExtractor::buildVocabulary() {
  printf("Building vocabulary...");
  random_shuffle(_unquantized.begin(), _unquantized.end());
  PidMat samples;
  int count = 0;
  for(int i = 0; i < _unquantized.size(); i++) {
    if(_labels[i] == 1) {
      samples.push_back(_unquantized[i]);
      count++;
    }
    if(count == _vocabularySamples) break;
  }
  cv::Mat l;
  cv::kmeans(samples, _vocabularySize, l, cv::TermCriteria(1, 1000, 0), 1, cv::KMEANS_RANDOM_CENTERS, _vocabulary);
  _index = new cv::flann::Index(_vocabulary, cv::flann::KMeansIndexParams());
  _vocabBuilt = true;
  printf("complete\n");
}

std::vector<PidMat> FeatureExtractor::quantize(std::vector<PidMat>& features) {
  std::vector<PidMat> qfeatures;
  for(int i = 0; i < features.size(); i++) {
    PidMat& feature = features[i];
    feature = quantize(feature);
    qfeatures.push_back(feature);
  }
  return qfeatures;
}

PidMat FeatureExtractor::quantize(PidMat& feature) {
  assert(_index != 0);
  PidMat quantized(_vocabularySize, 1);
  quantized = 0;
  cv::Mat indices, dists;
  _index->knnSearch(feature, indices, dists, 1, cv::flann::SearchParams());
  for(int r = 0; r < indices.rows; r++) {
    int index = indices.at<int>(r,0);
    quantized(index,0)++;
  }
  return quantized;
}

void FeatureExtractor::toYaml(YAML::Emitter& emitter) const {
  emitter << YAML::BeginSeq;
  for(int i = 0; i < _vocabulary.rows; i++) {
    emitter << YAML::BeginSeq;
    for(int j = 0; j < _vocabulary.cols; j++)
      emitter << _vocabulary(i, j);
    emitter << YAML::EndSeq;
  }
  emitter << YAML::EndSeq;
}

void FeatureExtractor::fromYaml(const YAML::Node& node) {
  if(!_isQuantized) return;
  assert(node.size() > 0);
  assert(node[0].size() > 0);
  _vocabulary = PidMat(node.size(), node[0].size());
  for(int i = 0; i < node.size(); i++) {
    const YAML::Node& vec = node[i];
    for(int j = 0; j < vec.size(); j++)
      vec[j] >> _vocabulary(i, j);
  }
  _index = new cv::flann::Index(_vocabulary, cv::flann::KMeansIndexParams());
  _vocabBuilt = true;
}

void FeatureExtractor::setQuantized(bool value) {
  _isQuantized = value;
} 

void FeatureExtractor::setLabels(const std::vector<int>& labels) {
  _labels = labels;
}

void FeatureExtractor::setQuantizePoint(int point) {
  _quantizePoint = point;
}

bool FeatureExtractor::isQuantized() const {
  return _isQuantized;
}

bool FeatureExtractor::vocabularyBuilt() const {
  return _vocabBuilt;
}

YAML::Emitter& operator<< (YAML::Emitter& emitter, const FeatureExtractor& extractor) {
  extractor.toYaml(emitter);
}

void operator>> (const YAML::Node& node, FeatureExtractor& extractor) {
  extractor.fromYaml(node);
}
