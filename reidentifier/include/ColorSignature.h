#ifndef COLOR_SIGNATURE_H
#define COLOR_SIGNATURE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Histogram.h"
#include <stdio.h>

#define FOREACH_SLICE(i) for(int i = 0; i < _slices; i++)

struct IllCorrect {
  double c1, c2, c3;
  IllCorrect() {
    c1 = c2 = c3 = 1.0;
  }
  IllCorrect(double c1, double c2, double c3) {
    this->c1 = c1; this->c2 = c2; this->c3 = c3;
  }
};

class ColorSignature {
  public:
    enum ColorSpace {
      RGB,
      YUV,
      LAB,
      HSV,
      HSL
    };
    ColorSignature(const cv::Mat&, cv::Rect, int slices = 4, ColorSpace = YUV);
    ColorSignature(const cv::Mat&, const cv::Mat&, cv::Rect, int slices = 4, ColorSpace = YUV);
    double distanceTo(const ColorSignature&) const;

    static bool ILLUMINATION_CORRECTION;
    
  private:
    void init(cv::Mat&, const cv::Mat&, cv::Rect, int, ColorSpace);
    Histogram getHistogram(cv::Mat&, const cv::Mat&, cv::Rect, IllCorrect);
    std::vector<IllCorrect> getCorrections() const;
    void convertColorSpace(cv::Mat&);
    std::vector<Histogram> _histograms;
    int _slices;
    ColorSpace _space;
};

#endif
