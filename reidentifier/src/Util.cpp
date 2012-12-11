#include "Util.h"

string Util::env(string name) {
  char* value = getenv(name.c_str());
  if(!value) {
    printf("WARNING: Environment variable %s is not set\n", name.c_str());
    throw -1;
  }
  return string(value);
}

vector<string> Util::getDirectoryImages(string directory) {
  vector<string> images;
  DIR *dir;
  struct dirent *entry;
  if(dir = opendir(directory.c_str())) {
    while(entry = readdir(dir)) {
      string name = entry->d_name;
      int index = name.find_last_of('.') + 1;
      if(index < name.length()) {
        string extension = name.substr(index, name.length() - index);
        if(
          extension == "bmp" || 
          extension == "jpg" || 
          extension == "png" ||
          extension == "pgm" ||
          extension == "gif"
        ) images.push_back(directory + "/" + name);
      }
    }
  }
  sort(images.begin(), images.end());
  return images;
}

vector<cv::Mat> Util::loadImages(string directory) {
  return Util::loadImages(Util::getDirectoryImages(directory));
}

cv::Mat Util::loadImage(string path) {
  cv::Mat image = cv::imread(path.c_str());
  return image;
}

vector<cv::Mat> Util::loadImages(vector<string> files) {
  vector<cv::Mat> images;
  BOOST_FOREACH(string file, files) {
    cv::Mat image = cv::imread(file.c_str());
    images.push_back(image);
  }
  return images;
}

string Util::getFileFromPath(string path) {
  string filename = path;
  size_t pos = path.find_last_of("/");
  if(pos != string::npos)
    filename.assign(path.begin() + pos + 1, path.end());
  return filename;
}

string Util::getDirectoryFromPath(string path) {
  string directory = "";
  size_t pos = path.find_last_of("/");
  if(pos != string::npos)
    directory.assign(path.begin(), path.begin() + pos);
  return directory;
}

LabelAssignments Util::loadLabels(string file) {
  ifstream fh(file.c_str());
  YAML::Parser parser(fh);
  YAML::Node node;
  parser.GetNextDocument(node);
  string directory = Util::getDirectoryFromPath(file);
  LabelAssignments labels;
  for(YAML::Iterator it = node.begin(); it != node.end(); ++it) {
    string filepath; int label;
    it.first() >> filepath;
    filepath = directory + "/" + filepath;
    it.second() >> label;
    labels[filepath] = label;
  }
  return labels;
}

TrainingImages Util::loadTrainingImages(string directory) {
  TrainingImages t;
  LabelAssignments labels = Util::loadLabels(directory + "/labels.yaml");
  BOOST_FOREACH(LabelAssignments::value_type& label, labels) {
    cv::Mat image = cvLoadImage(label.first.c_str());
    t.labels.push_back(label.second);
    t.images.push_back(image);
  }
  return t;
}

cv::Rect Util::correctBoundingBox(cv::Rect original, const cv::Mat& image) {
  if(original.x < 0) original.x = 0;
  if(original.y < 0) original.y = 0;
  if(original.x + original.width > image.cols) original.width = image.cols - original.x;
  if(original.y + original.height > image.rows) original.height = image.rows - original.y;
  return original;
}

bool Util::zip(string file) {
  string zipfile = file + ".gz";
  ifstream infile(file.c_str(), ios_base::in | ios_base::binary);
  if(!infile.good()) return false;
  boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
  in.push(boost::iostreams::gzip_compressor());
  in.push(infile);
  ofstream outfile(zipfile.c_str(), ios_base::out | ios_base::binary);
  boost::iostreams::copy(in, outfile);
  outfile.close();
  remove(file.c_str());
  return true;
}

bool Util::unzip(string file, string& unzipfile) {
  string zipfile = file + ".gz";
  unzipfile = Util::getFileFromPath(file);
  unzipfile = "/tmp/" + unzipfile;
  ifstream infile(zipfile.c_str(), ios_base::in | ios_base::binary);
  if(!infile.good()) return false;
  boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
  in.push(boost::iostreams::gzip_decompressor());
  in.push(infile);
  ofstream outfile(unzipfile.c_str(), ios_base::out | ios_base::binary);
  boost::iostreams::copy(in, outfile);
  outfile.close();
  return true;
}





