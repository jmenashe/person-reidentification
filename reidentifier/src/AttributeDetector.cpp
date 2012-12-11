#include "AttributeDetector.h"

AttributeDetector::AttributeDetector(std::string attribute, float weight, Parts::Location attLoc) {
  _attribute = attribute;
  _weight = weight;
  _attributeLoc = attLoc;

  _svm = new OcvSvm();  

  _threshold = 0.0;
  _hogExtractor = new HogFeatureExtractor();
  _siftExtractor = new SiftFeatureExtractor();
  _hogExtractor->setQuantized(false);
  _siftExtractor->setQuantized(true);
  _useHOG = true;
  _useSIFT = false;
}

AttributeDetector::~AttributeDetector() {
  delete _svm;
  delete _hogExtractor;
  delete _siftExtractor;
}

void AttributeDetector::load() {
  std::string directory = Util::env("SVM_MODELS");
  _svm->load(directory + "/" + _attribute + ".svm");
  
  std::string file = directory + "/" + _attribute + ".vcb";
  ifstream fh(file.c_str());
  if(!fh.good()) {
    printf("WARNING: Attribute detector file %s doesn't exist.\n", file.c_str());
    throw -1;
  }
  YAML::Parser parser(fh);
  YAML::Node node;
  parser.GetNextDocument(node);
  assert(node.size() > 0);
  int i = 0;
  if(_useSIFT) {
    node[i++] >> *_siftExtractor;
  }
  if(_useHOG) {
    node[i++] >> *_hogExtractor;
  }
  fh.close();
}

void AttributeDetector::save() {
  std::string directory = Util::env("SVM_MODELS");
  _svm->save(directory + "/" + _attribute + ".svm");
  
  std::string file = directory + "/" + _attribute + ".vcb";
  ofstream fh(file.c_str());
  YAML::Emitter emitter;
  emitter << YAML::BeginSeq;
  if(_useSIFT)
    emitter << *_siftExtractor;
  if(_useHOG)
    emitter << *_hogExtractor;
  emitter << YAML::EndSeq;
  fh << emitter.c_str();
  fh.close();
}

void AttributeDetector::train(const std::vector<std::string>& imageFiles, const std::vector<int>& labels) {
  std::vector<cv::Rect> boxes;
  for(int i = 0 ; i < imageFiles.size(); i++)
    boxes.push_back(cv::Rect(0,0,1000000,1000000));
  train(imageFiles, boxes, labels);
}

void AttributeDetector::train(const std::vector<std::string>& imageFiles, const std::vector<cv::Rect>& boxes, const std::vector<int>& labels) {
  int maxFile = 1000;
  if(maxFile >= imageFiles.size()) maxFile = imageFiles.size() - 1;
  if(_useHOG) {
    _hogExtractor->setQuantizePoint(maxFile);
    _hogExtractor->setLabels(labels);
  }
  if(_useSIFT) {
    _siftExtractor->setQuantizePoint(maxFile);
    _siftExtractor->setLabels(labels);
  }
  for(int i = 0; i < imageFiles.size(); i++) {
    std::string file = imageFiles[i];
    // Use this function to avoid memory leaks
    IplImage* imageP = cvLoadImage(file.c_str());
    cv::Mat image = imageP;
    cv::Rect box = boxes[i];
    box = Util::correctBoundingBox(box, image);
    cv::Mat roi = image(box);
    cv::Mat gray(roi.size(), CV_8U);
    if(roi.channels() != 1)
      cv::cvtColor(roi, gray, CV_RGB2GRAY, 1);
    else
        gray = roi.clone();
    if(_useSIFT) _siftExtractor->processImage(gray);
    if(_useHOG) _hogExtractor->processImage(gray);
    cvReleaseImage(&imageP);
  }
  FeatureSet fset;
  if(_useSIFT) { 
    fset.push_back(_siftExtractor->getFeatures());
    _siftExtractor->clearFeatures();
  }
  if(_useHOG) { 
    fset.push_back(_hogExtractor->getFeatures());
    _hogExtractor->clearFeatures();
  }
  std::vector<PidMat> combined = FeatureExtractor::combine(fset);
  assert(combined.size() == labels.size());
  _svm->train(combined, labels);
}

void AttributeDetector::normalize(std::vector<PidMat>& features) {
  BOOST_FOREACH(PidMat& feature, features) {
    normalize(feature);
  }
}

void AttributeDetector::normalize(PidMat& feature) {
  float max = 0;
  for(int i = 0; i < feature.rows; i++) {
    for(int j = 0; j < feature.cols; j++) {
      float temp = feature(i,j);
      if(max < temp) max = temp;
    }
  }
  for(int i = 0; i < feature.rows; i++)
    for(int j = 0; j < feature.cols; j++)
      feature(i,j) /= max;
}

bool AttributeDetector::hasAttribute(cv::Mat& image, cv::Rect box) {
  float prediction;
  return hasAttribute(image, box, prediction);
}

bool AttributeDetector::hasAttribute(cv::Mat& image, cv::Rect box, float& prediction) {
    cv::Mat roi = image(box);
    if(roi.channels() != 1)
      cv::cvtColor(roi, roi, CV_RGB2GRAY, 1);
    PidMat feature;
    if(_useSIFT) {
      assert(!_siftExtractor->isQuantized() || _siftExtractor->vocabularyBuilt());
      feature.push_back(_siftExtractor->extractFeature(roi));
    }
    if(_useHOG) {
      assert(!_hogExtractor->isQuantized() || _hogExtractor->vocabularyBuilt());
      feature.push_back(_hogExtractor->extractFeature(roi));
    }
    prediction = _svm->predict(feature);
    if(prediction > _threshold)
        return true;
    else
        return false;
}

float AttributeDetector::getWeight() {
  return _weight;
}

Parts::Location AttributeDetector::getDetectLoc()
{
    return _attributeLoc;
}

std::string AttributeDetector::getAttribute() {
  return _attribute;
}
