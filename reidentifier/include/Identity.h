#ifndef IDENTITY_H
#define IDENTITY_H

#include "ColorSignature.h"
#include "FaceDescriptor.h"

class Identity {
  public:
    int id;
    FaceDescriptor face; // Not being used right now
    std::vector<std::vector<bool> > attributes;
    std::vector<std::vector<ColorSignature> > signatures;
    std::vector<cv::Mat> faces;
};

#endif
