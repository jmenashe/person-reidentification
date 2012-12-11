#include "PartDetector.h"
#include "PersonDetector.h"

//#include <boost/threadpool.hpp>

//typedef boost::threadpool::pool tpool;

std::string IMAGES_DIR = std::string(Util::env("HOME")) + "/attributesData/training";
//std::string IMAGES_DIR = "/mnt/lab_images";
std::string PARTS_DIR = std::string(Util::env("HOG_MODELS"));
//std::string SAVE_PATH = Util::env("HOME") + "/images/lab";
std::string SAVE_PATH = Util::env("REIDENTIFIER_DIR") + "/SaveImages";

void processDirectory(std::string source_path, std::string save_path) {
    PersonDetector person;
    PartDetector personSegments;

//    assert(person.load(PARTS_DIR)==true);
    assert(person.load(PARTS_DIR)==true);
    printf("Loaded Person Model\nLoading Images...\n");
    std::vector<std::string> images = Util::getDirectoryImages(source_path);
    printf("%d Images Loaded from %s\n",images.size(),IMAGES_DIR.c_str());
    int imageNum = 1;
    BOOST_FOREACH(std::string file, images)
    {
        printf("Processing %s\n", file.c_str());
        cv::Mat image = Util::loadImage(file);
        std::vector<cv::Rect> People;
        person.detect(image,People);
//        People.push_back(cv::Rect(0,0,image.cols,image.rows));//Hack for viper dataset disable otherwise
        if(People.size()!=0)
        {
            int personCount = 1;
            BOOST_FOREACH(cv::Rect roi,People)
            {
                stringstream ss;
                ss << setw(4) << setfill('0') << imageNum;

                string imName = ss.str();

                ss.seekp(0);
                ss.seekg(0);
                ss.str("");
                ss.clear();
                ss << setw(2) << setfill('0') <<personCount;

                string personNum = ss.str();
                cv::Mat roiImage = image(roi);
                cv::imwrite(save_path + "/" +  imName+"_"+personNum+".png",roiImage);

                Parts segments = personSegments.getAllSegments(image,roi);

                if(segments.seen[Parts::Face])
                {
                    cv::Mat partImage = roiImage(segments.locations[Parts::Face]);
                    cv::imwrite(save_path+"/"+imName+"_"+personNum+"_ff"+".png",partImage);
                }

                if(segments.seen[Parts::FaceProfile])
                {
                    cv::Mat partImage = roiImage(segments.locations[Parts::FaceProfile]);
                    cv::imwrite(save_path+"/"+imName+"_"+personNum+"_fp"+".png",partImage);
                }

                if(segments.seen[Parts::UpperBody])
                {
                    cv::Mat partImage = roiImage(segments.locations[Parts::UpperBody]);
                    cv::imwrite(save_path+"/"+imName+"_"+personNum+"_ub"+".png",partImage);
                }

                if(segments.seen[Parts::LowerBody])
                {
                    cv::Mat partImage = roiImage(segments.locations[Parts::LowerBody]);
                    cv::imwrite(save_path+"/"+imName+"_"+personNum+"_lb"+".png",partImage);
                }
                personCount++;
            }
        }
        imageNum++;
    }
}
int main(int argc, char **argv) {
  {
//    tpool pool(4);
//    boost::function<void ()> f;
//    f = boost::bind(processDirectory, IMAGES_DIR + "/monday/axis9", SAVE_PATH + "/monday_axis9");
//    pool.schedule(f);
//    f = boost::bind(processDirectory, IMAGES_DIR + "/monday/axis25", SAVE_PATH + "/monday_axis25");
//    pool.schedule(f);
//    f = boost::bind(processDirectory, IMAGES_DIR + "/wednesday/axis9", SAVE_PATH + "/wednesday_axis9");
//    pool.schedule(f);
//    f = boost::bind(processDirectory, IMAGES_DIR + "/wednesday/axis25", SAVE_PATH + "/wednesday_axis25");
//    pool.schedule(f);
      processDirectory(IMAGES_DIR, SAVE_PATH);
  }
  return 0;
}
