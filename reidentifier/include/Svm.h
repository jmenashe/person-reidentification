#ifndef SVM_H
#define SVM_H

#include <boost/foreach.hpp>
#include <stdio.h>

#include "Util.h"
#include "Typedefs.h"


class Svm {
  public:
    virtual void train(const std::vector<PidMat>&,const std::vector<int>&) = 0;
    virtual float predict(const PidMat& feature) = 0;
    virtual void save(std::string) = 0;
    virtual void load(std::string) = 0;
};
#endif
