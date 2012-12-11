#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>

#define HIST_BIN1 4
#define HIST_BIN2 4
#define HIST_BIN3 4
#define HIST_INTERVAL1 (256.0 / HIST_BIN1)
#define HIST_INTERVAL2 (256.0 / HIST_BIN2)
#define HIST_INTERVAL3 (256.0 / HIST_BIN3)
#define HISTOGRAM_BINS (HIST_BIN1 * HIST_BIN2 * HIST_BIN3)

#define HIST_BIN_VALUE(x,n) ((int)(n == 1 ? (x / HIST_INTERVAL1) : n == 2 ? (x / HIST_INTERVAL2) : (x / HIST_INTERVAL3)))
#define HIST_BIN_INDEX(x,y,z) (HIST_BIN_VALUE(x,1) + HIST_BIN1 * HIST_BIN_VALUE(y,2) + HIST_BIN1 * HIST_BIN2 * HIST_BIN_VALUE(z,3))

typedef cv::Vec3b Color;

class Histogram {
  public:
    Histogram() {
      for(int i = 0; i < HISTOGRAM_BINS; i++)
        bins.push_back(0);
    }
    std::vector<int> bins;
};

#endif
    
  
