#include "PartDetector.h"
#include "Util.h"
#include "HogFeatureExtractor.h"
//using namespace std;
//using namespace cv;
#include <sstream>
#include <fstream>

std::string TRAINING_DIR_1 = std::string(Util::env("HOME")) + "/yaleFaces/train2";
std::string TESTING_DIR_1 = std::string(Util::env("HOME")) + "/yaleFaces/test2";

int main(int argc, char **argv) {
    PartDetector fTrain;
    std::vector<int> trainLabels;
    std::vector<int> testLabels;
    printf("Loading Train Images...\n");
    std::vector<cv::Mat> fTrainingImages = Util::loadImages(TRAINING_DIR_1);
    for(int i = 1;i<=fTrainingImages.size();i++)
    {
        int label = int(ceil((double)i/(double)1));
        cv::imwrite("pl.jpg",fTrainingImages[i-1]);
        trainLabels.push_back(label);
    }

    printf("Loading Test Images...\n");
    std::vector<cv::Mat> fTestingImages = Util::loadImages(TESTING_DIR_1);
    for(int i = 1;i<=fTestingImages.size();i++)
    {
        if(i<10)
        {
            testLabels.push_back(1);
            continue;
        }
        testLabels.push_back(int(ceil((double)(i)/(double)10)));
    }
    fTrain.TrainFaces(fTrainingImages,trainLabels);
    std::ofstream file;
//    for(double thresh = 5000;thresh<=15000;thresh+=1000)
//    {
        stringstream ss;
        file.open("faceROneTrain.txt",ios::out|ios::app);
        int i = 0;
        BOOST_FOREACH(cv::Mat im,fTestingImages)
        {
//        cv::imwrite("please.png",im);
//        int result = fTrain.recogFace(im);
            int result;
            double dist;
            fTrain.recogAnalysisFace(im,10,dist,result);
            printf("Predicted: %d Actual: %d\n",result,testLabels[i]);
            int act = ((testLabels[i]==20)?(-1):(testLabels[i]));
            i++;
//            ss<<act<<" "<<result<<" "<<setprecision(2)<<fixed<<dist<<" "<<setprecision(2)<<fixed<<thresh<<"\n";
            ss<<act<<" "<<result<<" "<<setprecision(2)<<fixed<<dist<<"\n";
        }
        file<<ss.str().c_str();
        file.close();
//    }
//    file.close();
    return 0;
}


