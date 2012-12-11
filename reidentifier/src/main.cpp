#include "Util.h"
#include "Identifier.h"
#include "PersonDetector.h"

#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>
#include <boost/foreach.hpp>
#include <boost/threadpool.hpp>
typedef boost::threadpool::pool tpool;

#include <map>

using namespace std;

string HOME_DIR(Util::env("HOME"));
string HOG_DIR(Util::env("HOG_MODELS"));

void shuffleSets(vector<string>& train, vector<string>& test, int samples) {
  assert(train.size() == test.size());
  vector<int> indexes;
  for(int i = 0; i < train.size(); i++)
    indexes.push_back(i);
  random_shuffle(indexes.begin(), indexes.end());
  vector<string> trainShuffle, testShuffle;
  for(int i = 0; i < samples; i++) {
    trainShuffle.push_back(train[indexes[i]]);
    testShuffle.push_back(test[indexes[i]]);
  }
  train = trainShuffle;
  test = testShuffle;
}

vector<int> reg(Identifier& identifier, vector<string> images) {
  vector<int> ids;
  for(int i = 0; i < images.size(); i++) {
    cv::Mat image = Util::loadImage(images[i]);
    int id = identifier.registerPerson(image);
    ids.push_back(id);
  }
  //identifier.updateFaceRecognizer();
  return ids;
}


//vector<int> viperRegister(Identifier& identifier) {
  //vector<cv::Mat> images = Util::loadImages(HOME_DIR + "/images/viperA");
  //vector<int> identities; //need to read identities from training data or just geenrate new one here instead of the identifier class

  //int id = 1; //hack

  //for(int i = 0; i < 20;i++){ //images.size(); i++) {
      //identities.push_back(id);
      //identifier.registerPerson(images[i],identities[i]);
      //if((i+1)%4==0)
              //id++;
  //}
  //bool trainResult = identifier.updateFaceRecognizer();
  //return identities;
//}

vector<int> viperTest(Identifier& identifier, vector<string> images) {
  vector<int> ids;
  for(int i = 0; i < images.size(); i++) {
    cv::Mat image = Util::loadImage(images[i]);
    int id = identifier.identifyPerson(image);
    ids.push_back(id);
  }
  return ids;
}

void runTrial(vector<string> regImages, vector<string> testImages, int samples, int& success) {
  success = 0;
  Identifier identifier;
  shuffleSets(regImages, testImages, samples);
  vector<int> registrations = reg(identifier, regImages);
  vector<int> tests = viperTest(identifier, testImages);
  assert(registrations.size() == tests.size());
  for(int i = 0; i < registrations.size(); i++) {
    if(registrations[i] == tests[i]) success++;
  }
}
  
int main(int argc, char* argv[]) {
  Identifier identifier;
  string ofile = Util::env("REIDENTIFIER_DIR") + "/results/main/success_rates_viper.csv";
  ofstream output(ofile.c_str());
  //vector<string> regImages = Util::getDirectoryImages(HOME_DIR + "/images/viperA");
  //vector<string> testImages = Util::getDirectoryImages(HOME_DIR + "/images/viperB");
  vector<string> regImages = Util::getDirectoryImages(Util::env("REIDENTIFIER_DIR") + "/lab_images/train");
  vector<string> testImages = Util::getDirectoryImages(Util::env("REIDENTIFIER_DIR") + "/lab_images/test");
  int trials = 10;
  for(int samples = 1; samples <= 30; samples++) {
    int success[trials];
    //for(int i = 0; i < trials; i++)
      //runTrial(regImages, testImages, samples, success[i]);
    {
      tpool pool(4);
      boost::function<void ()> f;
      for(int i = 0; i < trials; i++) {
        f = boost::bind(runTrial, regImages, testImages, samples, boost::ref(success[i]));
        pool.schedule(f);
      }
    }
    double sum = 0;
    for(int i = 0; i < trials; i++)
      sum += success[i];
    double successRate = sum / (samples * trials);
    double failRate = 1 - successRate;
    printf("Success: %2.3f, Fail: %2.3f, Samples: %i\n", successRate, failRate, samples);
    output << samples << "," << successRate << "," << failRate << "\n";
  }
  output.close();
  return 0;
}
