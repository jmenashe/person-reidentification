#include <boost/regex.hpp>
#include <boost/foreach.hpp>
#include "Util.h"
#include "AttributeDetector.h"
#include "Typedefs.h"
#include "OcvSvm.h"
#include "FeatureExtractor.h"
#include <opencv2/core/core.hpp>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <boost/threadpool.hpp>

typedef boost::threadpool::pool tpool;

using namespace std;

string RESULTS_DIR = string(Util::env("REIDENTIFIER_DIR")) + "/results/a_detections";

//string TRAINING_DIR = string(Util::env("HOME")) + "/attributesData/simple";
//string TESTING_DIR = string(Util::env("HOME")) + "/attributesData/simple";
string TRAINING_DIR = string(Util::env("HOME")) + "/attributesData/training";
string TESTING_DIR = string(Util::env("HOME")) + "/attributesData/testing";

string CALTECH[4] = {
  "airplanes",
  "cars",
  "faces",
  "motorbikes"
};

string ATTRIBUTES[9] = {
  "isMale",
  "hasLongHair",
  "hasGlasses",
  "hasHat",
  "hasTShirt",
  "hasLongSleeves",
  "hasShorts",
  "hasJeans",
  "hasLongPants"
};

// from precision scores
float WEIGHTS[9] = {
  0.70,
  0.00,
  0.67,
  0.00,
  0.54,
  0.76,
  0.31,
  0.00,
  0.00
};

enum Attribute {
  IsMale,
  HasLongHair,
  HasGlasses,
  HasHat,
  HasTShirt,
  HasLongSleeves,
  HasShorts,
  HasJeans,
  HasLongPants,
  NumAttributes
};

enum CalAttribute {
  IsPlane,
  IsMotorbike,
  IsCar,
  IsFace,
  NumCalAttributes
};

struct Stats {
  Stats() {
    precision = recall = accuracy = 0;
    tp = fp = tn = fn = 0;
  }
  double precision;
  double recall;
  double accuracy;
  int tp;
  int fp;
  int tn;
  int fn;
};

class TrainingLabel {
  public:
    int attributes[NumAttributes];
    std::string image;
    double x,y,w,h;
    cv::Rect bb;
    bool valid;
};

class Dataset {
  public:
    vector<int> labels;
    vector<string> files;
    vector<cv::Rect> boxes;
};

vector<TrainingLabel> readLabels(string file) {
  ifstream infile(file.c_str());
  string line;
  vector<TrainingLabel> labels;
  while(getline(infile, line)) {
    stringstream ss(line);
    TrainingLabel label;
    label.valid = true;
    ss >> label.image;
    ss >> label.x;
    label.valid = label.valid && !ss.fail();
    ss >> label.y;
    label.valid = label.valid && !ss.fail();
    ss >> label.w;
    label.valid = label.valid && !ss.fail();
    ss >> label.h;
    label.valid = label.valid && !ss.fail();
    label.bb = cv::Rect((int)label.x, (int)label.y, (int)label.w, (int)label.h);
    for(int i = 0; i < NumAttributes; i++)
      ss >> label.attributes[i];
    labels.push_back(label);
  }
  infile.close();
  return labels;
}

Dataset buildCaltechDataset(CalAttribute att, bool train) {
  string suffix = train ? "_train" : "_test";
  Dataset dataset;
  for(int i = 0; i < NumCalAttributes; i++) {
    string directory = Util::env("HOME") + "/caltech";
    vector<string> files = Util::getDirectoryImages(directory + "/" + CALTECH[i] + suffix);
    for(int j = 0; j < files.size(); j++) {
      dataset.files.push_back(files[j]);
      dataset.boxes.push_back(cv::Rect(0,0,10000,10000));
      dataset.labels.push_back(i == att ? 1 : -1);
    }
  }
  return dataset;
}

Dataset buildPoseletsDataset(Attribute att, bool train) {
  Dataset dataset;
  string labelFile = train ? TRAINING_DIR : TESTING_DIR;
  labelFile += "/labels.txt";
  vector<TrainingLabel> labels = readLabels(labelFile);
  vector<string> files = Util::getDirectoryImages(train ? TRAINING_DIR : TESTING_DIR);
  for(int i = 0; i < labels.size(); i++) {
    TrainingLabel label = labels[i];
    if(!label.valid || label.attributes[att] == 0 || label.bb.height * label.bb.width <= 0) continue;
    dataset.labels.push_back(label.attributes[att] == 1 ? 1 : -1);
    dataset.boxes.push_back(label.bb);
    dataset.files.push_back(files[i]);
  }
  return dataset;
}
  
Dataset buildPoseletsDataset(Attribute att) {
  Dataset trainDataset = buildPoseletsDataset(att, true);
  Dataset testDataset = buildPoseletsDataset(att, false);
  for(int i = 0; i < testDataset.labels.size(); i++) {
    trainDataset.labels.push_back(testDataset.labels[i]);
    trainDataset.boxes.push_back(testDataset.boxes[i]);
    trainDataset.files.push_back(testDataset.files[i]);
  }
  return trainDataset;
}

void train(AttributeDetector* ad, Dataset& dataset) {
  printf("Beginning training\n");
  ad->train(dataset.files, dataset.boxes, dataset.labels);
}

Stats getStats(AttributeDetector* ad, Dataset& dataset) {
  Stats stats;
  for(int i = 0; i  < dataset.files.size(); i++) {
    string file = dataset.files[i];
    IplImage* imageP = cvLoadImage(file.c_str());
    cv::Mat image = imageP;
    cv::Rect box = dataset.boxes[i];
    box = Util::correctBoundingBox(box, image);
    float value = 0;
    bool result;
    if(ad) result = ad->hasAttribute(image,box,value);
    else result = rand() % 2;
    if(dataset.labels[i] == 1) {
      if(result) stats.tp++;
      else stats.fn++;
    }
    else {
      if(result) stats.fp++;
      else stats.tn++;
    }
    stats.precision = (double)stats.tp / (stats.tp + stats.fp);
    stats.recall = (double)stats.tp / (stats.tp + stats.fn);
    stats.accuracy = (double)(stats.tp + stats.tn) / (stats.tp + stats.fp + stats.tn + stats.fn);
    cvReleaseImage(&imageP);
  }
  return stats;
}

void writeStats(ostream& out, Stats s) {
  out << 
    "TP: " << s.tp << 
    ", FP: " << s.fp << 
    ", TN: " << s.tn << 
    ", FN: " << s.fn << "\n";
  out << 
    "Accuracy " << s.accuracy << 
    ", Precision: " << s.precision << 
    ", Recall: " << s.recall << "\n";
  printf("Precision: %2.2f, Recall: %2.2f, Accuracy: %2.2f\n", s.precision, s.recall, s.accuracy);
}
  
void test(AttributeDetector* ad, Dataset& dataset) {
  ofstream results((RESULTS_DIR + "/" + ad->getAttribute() + ".csv").c_str());
  Stats s = getStats(ad, dataset);
  results << "Detector:\n";
  writeStats(results, s);
  s = getStats(0, dataset);
  results << "Random:\n";
  writeStats(results, s);
  results.close();
}

void trainTest(AttributeDetector* ad, Attribute att) {
  Dataset dataset = buildPoseletsDataset(att, true);
  train(ad, dataset);
  ad->save();
  dataset = buildPoseletsDataset(att, false);
  test(ad, dataset);
  delete ad;
}


int main(int argc, char **argv) {
  string suffix = "_linear_hog_only";
  srand(time(NULL));
  {
    tpool pool(4);
    int start = 0;
    int end = NumAttributes;
    boost::function<void ()> f;
    for(int i = start; i < end; i++) {
      printf("Training %s\n", ATTRIBUTES[i].c_str());
      AttributeDetector* ad = new AttributeDetector(ATTRIBUTES[i] + suffix, WEIGHTS[i]);
      f = boost::bind(trainTest, ad, (Attribute)i);
      pool.schedule(f);
    }
  }
  return 0;
}
