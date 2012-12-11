#include "ColorSignature.h"

bool ColorSignature::ILLUMINATION_CORRECTION = false;

ColorSignature::ColorSignature(const cv::Mat& image, cv::Rect box, int slices, ColorSpace space) {
  cv::Mat mask;
  cv::Mat clone = image.clone();
  init(clone, mask, box, slices, space);
}

ColorSignature::ColorSignature(const cv::Mat& image, const cv::Mat& mask, cv::Rect box, int slices, ColorSpace space) {
  cv::Mat clone = image.clone();
  init(clone, mask, box, slices, space);
}

void ColorSignature::init(cv::Mat& image, const cv::Mat& mask, cv::Rect box, int slices, ColorSpace space) {
  _slices = slices;
  _space = space;
  convertColorSpace(image);
  std::vector<IllCorrect> corrections = getCorrections();
  for(int c = 0; c < corrections.size(); c++) {
    FOREACH_SLICE(i) {
      int x = box.x;
      int y = box.y + i * box.height / _slices;
      int width = box.width;
      int height = box.height / _slices;
      cv::Rect slice(x,y,width,height);
      std::vector<IllCorrect> corrections = getCorrections();
      Histogram histogram = getHistogram(image, mask, slice, corrections[c]);
      _histograms.push_back(histogram);
    }
  }
}

void ColorSignature::convertColorSpace(cv::Mat& image) {
  cv::Mat target;
  switch(_space) {
    case RGB: return;
    case YUV: cv::cvtColor(image, target, CV_BGR2YCrCb); break;
    case HSL: cv::cvtColor(image, target, CV_BGR2HLS); break;
    case LAB: cv::cvtColor(image, target, CV_BGR2Lab); break;
    case HSV: cv::cvtColor(image, target, CV_BGR2HSV); break;
  }
  image = target;
}
    

std::vector<IllCorrect> ColorSignature::getCorrections() const {
  std::vector<IllCorrect> corrections;
  if(!ILLUMINATION_CORRECTION) {
    corrections.push_back(IllCorrect());
    return corrections;
  }
  double c1Min = .8, c2Min = .8, c3Min = .8;
  switch(_space) {
    case HSL: c1Min = c3Min = 1.0; break;
    case YUV:
    case LAB: c2Min = c3Min = 1.0; break;
    case HSV: c1Min = c2Min = 1.0; break;
  }
  for(double c1 = c1Min; c1 <= 1; c1 += .1)
    for(double c2 = c2Min; c2 <= 1; c2 += .1)
      for(double c3 = c3Min; c3 <= 1; c3 += .1)
        corrections.push_back(IllCorrect(c1,c2,c3));
  return corrections;
}

Histogram ColorSignature::getHistogram(cv::Mat& image, const cv::Mat& mask, cv::Rect slice, IllCorrect correction) {
  Histogram histogram;
  int w = slice.width / 3;
  int h = slice.height;
  int cx = slice.x + slice.width / 2;
  for(int x = cx - w; x < cx + w; x++) {
    for(int y = slice.y; y < slice.y + h; y++) {
      // If the mask is uninitialized then ignore it
      bool allow = mask.empty() || mask.at<bool>(y,x);
      if(!allow) continue;
      Color pixel = image.at<Color>(y,x);
      uchar c1 = pixel[0], c2 = pixel[1], c3 = pixel[2];
      c1 = (int)(c1 * correction.c1);
      c2 = (int)(c2 * correction.c2);
      c3 = (int)(c3 * correction.c3);
      int index = HIST_BIN_INDEX(c1,c2,c3);
      histogram.bins[index]++;
    }
  }
  return histogram;
}

double ColorSignature::distanceTo(const ColorSignature& other) const {
  std::vector<IllCorrect> corrections = getCorrections();
  double minDist = -1;
  for(int c1 = 0; c1 < corrections.size(); c1++) {
    for(int c2 = 0; c2 < corrections.size(); c2++) {
      double distance = 0;
      FOREACH_SLICE(i) {
        for(int j = 0; j < HISTOGRAM_BINS; j++) {
          double lbin = _histograms[c1 + i].bins[j], rbin = other._histograms[c2 + i].bins[j];
          if(lbin > 0 || rbin > 0) {
            distance += pow((lbin - rbin), 2) / (lbin + rbin);
          }
        }
      }
      if(minDist < 0 || minDist > distance)
        minDist = distance;
    }
  }
  return minDist;
}

