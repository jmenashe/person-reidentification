#ifndef DLIB_SVM_H
#define DLIB_SVM_H

#include "Svm.h"

#include "dlib/svm.h"

typedef dlib::matrix<double,0,1> Sample;
typedef dlib::linear_kernel<Sample> Kernel;
typedef dlib::decision_function<Kernel> DecFuncType;
typedef dlib::normalized_function<DecFuncType> FuncType;

class DlibSvm {
  public:
    DlibSvm();
    void train(const std::vector<PidMat>&,const std::vector<int>&);
    float predict(const PidMat& feature);
    void save(std::string);
    void load(std::string);

  private:
    DecFuncType _svm;
};
#endif
