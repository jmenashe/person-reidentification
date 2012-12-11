#include "LibsvmSvm.h"

LibsvmSvm::LibsvmSvm() {
  _param.svm_type = C_SVC;
  _param.kernel_type = LINEAR;
  _param.degree = 3;
  _param.gamma = 6;//0.01;  // 1/num_features
  _param.coef0 = 0;
  _param.nu = 0.01;
  _param.cache_size = 1024;
  _param.C = 1;
  _param.eps = 1e-3;
  _param.p = 0.1;
  _param.shrinking = 1;
  _param.probability = 0;
  _param.nr_weight = 0;
  _param.weight_label = NULL;
  _param.weight = NULL;
}

void LibsvmSvm::train(const std::vector<PidMat>& features,const std::vector<int>& labels) {
  assert(features.size() > 0 && labels.size() == features.size());
  int featureCount = features.size();
  int featureLength = features[0].rows;
  svm_problem prob;
  prob.l = featureCount;
  prob.y = new double[featureCount];
  prob.x = new struct svm_node*[featureCount];
  for(int i = 0; i < featureCount; i++) {
    prob.x[i] = new struct svm_node[featureLength + 1];
    for(int j = 0; j < featureLength; j++) {
      svm_node node;
      node.index = j + 1;
      node.value = features[i](j,0);
      prob.x[i][j] = node;
    }
    prob.x[i][featureLength].index = -1;
    prob.y[i] = labels[i];
  }
  _model = svm_train(&prob, &_param);
}

float LibsvmSvm::predict(const PidMat& feature) {
  int featureLength = feature.rows;
  struct svm_node nodes[featureLength + 1];
  for(int i = 0; i < featureLength; i++) {
    nodes[i].value = feature(i,0);
    nodes[i].index = i + 1;
  }
  nodes[featureLength].index = -1;
  double value;
  svm_predict_values(_model, nodes, &value);
  return value;
}

void LibsvmSvm::save(std::string file) {
  svm_save_model(file.c_str(), _model);
  Util::zip(file);
}

void LibsvmSvm::load(std::string file) {
  std::string zipfile;
  if(!Util::unzip(file, zipfile)) {
    printf("Zipped model file doesn't exist.\n");
    throw -1;
  }
  _model = svm_load_model(zipfile.c_str());
  if(!_model) {
    printf("SVM Model file doesn't exist.\n");
    throw -1;
  }
  assert(_model != 0);
  //printf("Loaded %s\n", file.c_str());
}

