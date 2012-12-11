#include "PartDetector.h"

PartDetector::PartDetector() {
  _facesRecogModel = "face_recognizer";
  _fRecog = cv::createEigenFaceRecognizer();
}

Parts PartDetector::getAllSegments(const cv::Mat& image,const cv::Rect& person) {
  Parts parts;
  cv::Mat PersonOnly = image(person);
  cv::Mat ImageResized;
  ImageResized = PersonOnly;

  findFace(ImageResized, parts);
  findUpperLowerBody(ImageResized, parts);
  findFaceProfile(ImageResized, parts);
  parts.locations[Parts::WholeBody] = person;
  parts.seen[Parts::WholeBody] = true;
  return parts;
}

void PartDetector::findFace(const cv::Mat& image, Parts& parts)
{
    std::vector<cv::Rect> face;
    std::vector<std::string> faceModels;
    faceModels.push_back(Util::env("HAAR_MODELS") + "/frontalFace10/haarcascade_frontalface_default.xml");
    faceModels.push_back(Util::env("HAAR_MODELS") + "/frontalFace10/haarcascade_frontalface_alt2.xml");
    faceModels.push_back(Util::env("HAAR_MODELS") + "/frontalFace10/haarcascade_frontalface_alt.xml");
    faceModels.push_back(Util::env("HAAR_MODELS") + "/frontalFace10/haarcascade_frontalface_alt_tree.xml");

    for(int i = 0;i<faceModels.size();i++)
    {
        cv::CascadeClassifier detector(faceModels[i]);
        assert(!detector.empty());
        detector.detectMultiScale(image,face,1.1, 2, 0|CV_HAAR_SCALE_IMAGE
                                                      |CV_HAAR_FIND_BIGGEST_OBJECT
                                                    , cv::Size(5, 5));
        if(!face.empty())
            break;
    }
    parts.seen[Parts::Face] = !face.empty();
    if(!face.empty())
      parts.locations[Parts::Face] = face[0];
}

void PartDetector::findUpperLowerBody(const cv::Mat& image, Parts& parts) {
    std::vector<cv::Rect> upperBody;
    std::vector<cv::Rect> lowerBody;
    std::vector<std::string> UpperBodyParam;
    UpperBodyParam.push_back(Util::env("HAAR_MODELS") + "/body10/haarcascade_upperbody.xml");
    UpperBodyParam.push_back(Util::env("HAAR_MODELS") + "/HS.xml");
    std::string LowerBodyParam = Util::env("HAAR_MODELS") + "/body10/haarcascade_lowerbody.xml";

    for(int i = 0;i<UpperBodyParam.size();i++)
    {
        cv::CascadeClassifier UpperBodyDetector(UpperBodyParam[i]);
        assert(!UpperBodyDetector.empty());
        UpperBodyDetector.detectMultiScale(image,upperBody,1.1, 2, 0|CV_HAAR_SCALE_IMAGE
                                                      |CV_HAAR_FIND_BIGGEST_OBJECT
                                                    , cv::Size(5, 5));
        if(!upperBody.empty())
            break;
    }

    cv::CascadeClassifier LowerBodyDetector(LowerBodyParam);
        LowerBodyDetector.detectMultiScale(image,lowerBody,1.1, 2, 0|CV_HAAR_SCALE_IMAGE
                                                                |CV_HAAR_FIND_BIGGEST_OBJECT
                                                                , cv::Size(5,5));;
    parts.seen[Parts::UpperBody] = !upperBody.empty();
    parts.seen[Parts::LowerBody] = !lowerBody.empty();
    if(!upperBody.empty())
      parts.locations[Parts::UpperBody] = upperBody[0];
    if(!lowerBody.empty())
      parts.locations[Parts::LowerBody] = lowerBody[0];
}

void PartDetector::findFaceProfile(const cv::Mat& image, Parts& parts) {
    std::vector<cv::Rect> faceProfile;
    std::string FaceProfileParam = Util::env("HAAR_MODELS") + "/haarcascade_profileface.xml";
    cv::CascadeClassifier FaceProfileDetector(FaceProfileParam);
    FaceProfileDetector.detectMultiScale(image,faceProfile,1.1, 2, 0|CV_HAAR_SCALE_IMAGE
                                                                    |CV_HAAR_FIND_BIGGEST_OBJECT
                                                                    , cv::Size(5, 5));
    parts.seen[Parts::FaceProfile] = !faceProfile.empty();
    if(!faceProfile.empty())
      parts.locations[Parts::FaceProfile] = faceProfile[0];
}

void PartDetector::TrainFaces(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    std::vector<cv::Mat> grayImages;
    std::vector<cv::Mat> grayResized;
    convert2gray(images,grayImages);
    resizeImages(grayImages, grayResized);
    _fRecog->train(grayResized,labels);
}

void PartDetector::resizeImages(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& resizedImages)
{
    BOOST_FOREACH(cv::Mat image,images)
    {
//        cv::resize(image,image,cv::Size(20,20));
        resizedImages.push_back(image);
    }
}

void PartDetector::UpdateFaceModel(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
}

int PartDetector::recogFace(const cv::Mat& face) {
    cv::Mat grayImage;
    if(face.channels() != 1)
        cv::cvtColor(face, grayImage, CV_RGB2GRAY, 1);
    else
        grayImage = face.clone();
    cv::Mat imResized;
    cv::resize(grayImage,imResized,cv::Size(20,20));
    double d;
    int result;
    _fRecog->predict(grayImage,result,d);
    printf("\n%2.2f:%d\n",d,result);
    return(_fRecog->predict(grayImage));
}

void PartDetector::recogAnalysisFace(const cv::Mat& face,double thresh, double& d, int& result)
{
    cv::Mat grayImage;
    if(face.channels() != 1)
        cv::cvtColor(face, grayImage, CV_RGB2GRAY, 1);
    else
        grayImage = face.clone();
//    _fRecog->set("threshold",thresh);
    _fRecog->predict(grayImage,result,d);
}

void PartDetector::convert2gray(const std::vector<cv::Mat>& images,std::vector<cv::Mat>& grayImages)
{
    BOOST_FOREACH(cv::Mat image,images) {
        if(image.channels() != 1)
            cv::cvtColor(image, image, CV_RGB2GRAY, 1);
        grayImages.push_back(image);
    }
}
