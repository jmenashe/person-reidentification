#ifndef CV_SVM_H
#define CV_SVM_H

#include "Svm.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

class OcvSvm : public Svm {
  public:
    OcvSvm();
    void train(const std::vector<PidMat>&,const std::vector<int>&);
    float predict(const PidMat& feature);
    void save(std::string);
    void load(std::string);

  private:
    cv::SVM _cvsvm;
};
#endif
