#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include "FaceDescriptor.h"
#include "PersonDetector.h"
#include "Util.h"
#include "Parts.h"

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/contrib/contrib.hpp>

class PartDetector {

    const static int dImHeight = 60;
    const static int dImWidth = 60;
    std::string _facesRecogModel;
    cv::Ptr<cv::FaceRecognizer> _fRecog;
    void findFace(const cv::Mat&, Parts&);
    void findUpperLowerBody(const cv::Mat&, Parts&);
    void findFaceProfile(const cv::Mat&, Parts&);
    void convert2gray(const std::vector<cv::Mat>&,std::vector<cv::Mat>&);
    void resizeImages(const std::vector<cv::Mat>&,std::vector<cv::Mat>&);
public:
    PartDetector();
    Parts getAllSegments(const cv::Mat&,const cv::Rect&);
    void TrainFaces(const std::vector<cv::Mat>&, const std::vector<int>&);
    void UpdateFaceModel(const std::vector<cv::Mat>&, const std::vector<int>&); //LBP Only
    int recogFace(const cv::Mat&);
    void recogAnalysisFace(const cv::Mat&,double, double&, int&);
};

#endif

