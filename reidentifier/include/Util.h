#ifndef UTIL_H
#define UTIL_H

#include <dirent.h>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <boost/foreach.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <yaml-cpp/yaml.h>

using namespace std;

typedef map<string, int> LabelAssignments;

struct TrainingImages {
  vector<int> labels;
  vector<cv::Mat> images;
};

class Util {
  public:
    static vector<string> getDirectoryImages(string);
    static vector<cv::Mat> loadImages(vector<string>);
    static vector<cv::Mat> loadImages(string);
    static cv::Mat loadImage(string);
    static LabelAssignments loadLabels(string);
    static string getFileFromPath(string);
    static string getDirectoryFromPath(string);
    static TrainingImages loadTrainingImages(string directory);
    static cv::Rect correctBoundingBox(cv::Rect, const cv::Mat&);
    static void init();
    static string env(string);
    static bool zip(string);
    static bool unzip(string,string&);
};
#endif
