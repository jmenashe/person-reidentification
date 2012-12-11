#ifndef PARTS_H
#define PARTS_H

#include <opencv2/core/core.hpp>

class Parts {
  public:
    Parts() {
      memset(seen, false, NumLocations);
    }
    enum Location {
      UpperBody = 0,
      LowerBody,
      Face,
      FaceProfile,
      WholeBody,
      NumLocations
    };

    cv::Rect locations[NumLocations];
    bool seen[NumLocations];
};

#endif
