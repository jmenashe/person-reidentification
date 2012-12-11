#include "Util.h"
#include "ColorSignature.h"
#include <stdio.h>
#include <iostream>
#include <boost/foreach.hpp>
#include <fstream>
#include <boost/threadpool.hpp>
typedef boost::threadpool::pool tpool;

using namespace std;

string HOME_DIR(Util::env("HOME"));
string VIPER_A = HOME_DIR + "/images/viperA";
string VIPER_B = HOME_DIR + "/images/viperB";
class ScoredImage {
  public:
    string testImage;
    string image;
    int testIdx;
    int idx;
    double score;
    bool correct;
    static bool sort(const ScoredImage& i1, const ScoredImage& i2) {
      return i1.score < i2.score;
    }
};

vector<ScoredImage> testColorSignatures(int slices, ColorSignature::ColorSpace space) {
  vector<string> testImageFiles = Util::getDirectoryImages(VIPER_A);
  vector<cv::Mat> testImages = Util::loadImages(testImageFiles);
  vector<ScoredImage> simages;
  vector<string> imageFiles = Util::getDirectoryImages(VIPER_B);
  vector<cv::Mat> images = Util::loadImages(imageFiles);
  for(unsigned i = 0; i < testImages.size(); i++) {
    cv::Mat testImage = testImages[i];
    string testName = Util::getFileFromPath(testImageFiles[i]);
    ColorSignature testSignature(testImage, cv::Rect(0,0,testImage.cols,testImage.rows), slices, space);
    printf("checking image: %s\n", testName.c_str());
    for(unsigned j = 0; j < testImages.size(); j++) {
      cv::Mat image = images[j];
      string name = Util::getFileFromPath(imageFiles[j]);
      ColorSignature signature(image, cv::Rect(0,0,image.cols,image.rows), slices, space);
      double distance = testSignature.distanceTo(signature);
      ScoredImage si;
      si.testImage = testName;
      si.image = name;
      si.correct = (j == i);
      si.score = distance;
      si.testIdx = i;
      si.idx = j;
      simages.push_back(si);
    }
  }
  return simages;
}

vector<int> rankScoredImages(vector<ScoredImage> simages) {
  vector<int> ranks;
  for(int i = 0; i < simages.size();) {
    int curIdx = simages[i].testIdx;
    vector<ScoredImage> ranked;
    for(int j = i; j < simages.size(); j++) {
      if(curIdx == simages[j].testIdx){
        ranked.push_back(simages[j]);
      }
      else {
        i = j;
        break;
      }
      if(j == simages.size() - 1) i = j + 1;
    }
    sort(ranked.begin(), ranked.end(), ScoredImage::sort);
    for(int j = 0; j < ranked.size(); j++) {
      if(ranked[j].idx == curIdx) {
        ranks.push_back(j);
        break;
      }
    }
  }
  return ranks;
}

void buildRankCurve(int slices, ColorSignature::ColorSpace space, string file) {
  ofstream output(file.c_str());
  vector<ScoredImage> simages = testColorSignatures(slices, space);
  vector<int> ranks = rankScoredImages(simages);
  vector<string> files = Util::getDirectoryImages(VIPER_A);
  int minRank = files.size() - 1, maxRank = 0;
  for(int i = 0; i < files.size(); i++) {
    if(minRank > ranks[i]) minRank = ranks[i];
    if(maxRank < ranks[i]) maxRank = ranks[i];
  }
  for(int r = minRank; r <= maxRank; r++) {
    int matches = 0;
    for(int i = 0; i < ranks.size(); i++)
      if(ranks[i] <= r)
        matches++;
    output << r << "," << matches << "\n";
  }
  output.close();
}

void runRankCurves() {
  ColorSignature::ILLUMINATION_CORRECTION = false;
  string mod = "4";
  int slices = 4;
  string dir = Util::env("REIDENTIFIER_DIR") + "/results/c_signatures/ranks/";
  {
    tpool pool(4);
    boost::function<void ()> f;
    f = boost::bind(buildRankCurve, slices, ColorSignature::YUV, dir + "yuv_" + mod + ".csv");
    pool.schedule(f);
    f = boost::bind(buildRankCurve, slices, ColorSignature::HSV, dir + "hsv_" + mod + ".csv");
    pool.schedule(f);
    f = boost::bind(buildRankCurve, slices, ColorSignature::LAB, dir + "lab_" + mod + ".csv");
    pool.schedule(f);
    f = boost::bind(buildRankCurve, slices, ColorSignature::HSL, dir + "hsl_" + mod + ".csv");
    pool.schedule(f);
    f = boost::bind(buildRankCurve, slices, ColorSignature::HSV, dir + "rgb_" + mod + ".csv");
    pool.schedule(f);
  }
}
    

void buildRocCurve(vector<ScoredImage> simages, string file) {
  ofstream output(file.c_str());
  int resolution = 100;
  double minScore = 1000000, maxScore = 0;
  for(int i = 0; i < simages.size(); i++) {
    double score = simages[i].score;
    if(minScore > score) minScore = score;
    if(maxScore < score) maxScore = score;
  }
  double step = (maxScore - minScore) / resolution;
  int tpMax = sqrt(simages.size());
  int fpMax = simages.size() - tpMax;
  for(double score = minScore; score <= maxScore; score += step) {
    int fp = 0;
    int tp = 0;
    for(int i = 0; i < simages.size(); i++) {
      if(simages[i].score <= score) {
        if(simages[i].correct) tp++;
        else fp++;
      }
    }
    double fpRate = (double)fp / fpMax;
    double tpRate = (double)tp / tpMax;
    output << fpRate << "," << tpRate << "\n";
    //cout << fpRate << "," << tpRate << "\n";
  }
  output.close();
}

void runTest(int slices, ColorSignature::ColorSpace space, string filename) {
  vector<ScoredImage> infos = testColorSignatures(slices, space);
  buildRocCurve(infos, filename);
}

void runRocCurves() {
  ColorSignature::ILLUMINATION_CORRECTION = true;
  for(int i = 1; i <= 5; i++) {
    printf("Running %i slices\n", i);
    int slices = i;
    stringstream ss;
    ss << "_" << slices << "_corrected";
    string mod = ss.str();
    string dir = Util::env("REIDENTIFIER_DIR") + "/results/c_signatures/roc/";
    {
      tpool pool(4);
      boost::function<void ()> f;
      f = boost::bind(runTest, slices, ColorSignature::YUV, dir + "yuv" + mod + ".csv");
      pool.schedule(f);
      f = boost::bind(runTest, slices, ColorSignature::HSV, dir + "hsv" + mod + ".csv");
      pool.schedule(f);
      f = boost::bind(runTest, slices, ColorSignature::HSL, dir + "hsl" + mod + ".csv");
      pool.schedule(f);
      f = boost::bind(runTest, slices, ColorSignature::LAB, dir + "lab" + mod + ".csv");
      pool.schedule(f);
      //f = boost::bind(runTest, slices, ColorSignature::RGB, dir + "rgb" + mod + ".csv");
      //pool.schedule(f);
    }
  }
}

int main(int argc, char *argv[]) {
  //runRankCurves();
  runRocCurves();
  return 0;
}
