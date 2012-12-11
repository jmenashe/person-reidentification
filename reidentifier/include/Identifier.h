#ifndef IDENTIFIER_H
#define IDENTIFIER_H

#define UNIDENTIFIED_PERSON 0

#include "Identity.h"
#include <vector>
#include "PartDetector.h"
#include "PersonDetector.h"
#include "AttributeDetector.h"
#include "ColorSignature.h"
#include <opencv2/core/core.hpp>

class Identifier {
  public:
    Identifier();
    int identifyPerson(cv::Mat&);
    int identifyPerson(cv::Mat&, cv::Rect);
    int registerPerson(cv::Mat&, int id = -1);
    int registerPerson(cv::Mat&, cv::Rect, int id = -1);
    bool updateFaceRecognizer();
  private:
    int recognizeFace(const cv::Mat&, cv::Rect);
    int identifySignature(ColorSignature&, std::vector<Identity>);
    
    std::vector<bool> getAttributes(cv::Mat&, cv::Rect, Parts& p);

    std::vector<Identity>  attributeMatches(std::vector<bool>);
    
    
    std::vector<Identity> _identities;    
    static std::vector<AttributeDetector*> _attributeDetectors;
    PersonDetector _personDetector;
    PartDetector _partDetector;
    std::vector<std::pair<std::pair<PidMat,PidMat>,int> > corrMatrices;
    void computeCorrelationAll();
    std::vector<std::pair<double,int> > likelyPersonID(std::vector<bool>);
    static bool pairCompare(const std::pair<double,int>&,const std::pair<double,int>&);
    std::vector<Identity> probabilisticMatches(std::vector<bool>);
    static const bool _useProbWeighting = false;

    static int __id; // Store the last id assignment so that they don't overlap
    
};

#endif
