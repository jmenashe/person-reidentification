#ifndef LIBSVM_SVM_H
#define LIBSVM_SVM_H

#include "Svm.h"
#include "libSvm.h"

class LibsvmSvm : public Svm {
  public:
    LibsvmSvm();
    void train(const std::vector<PidMat>&,const std::vector<int>&);
    float predict(const PidMat& feature);
    void save(std::string);
    void load(std::string);
  private:
    struct svm_model *_model;
    struct svm_parameter _param;
};
#endif
