#include "Identifier.h"

int Identifier::__id = 0;

std::vector<AttributeDetector*> Identifier::_attributeDetectors;

Identifier::Identifier() {
  if(_attributeDetectors.size() == 0) {
    _attributeDetectors.push_back(new AttributeDetector("isMale", 0.5624, Parts::WholeBody));
    _attributeDetectors.push_back(new AttributeDetector("hasLongHair", 0.6308, Parts::Face));
    _attributeDetectors.push_back(new AttributeDetector("hasGlasses", 0.6408, Parts::Face));
    _attributeDetectors.push_back(new AttributeDetector("hasHat", 0.7702, Parts::Face));
    _attributeDetectors.push_back(new AttributeDetector("hasTShirt", 0.6603));
    _attributeDetectors.push_back(new AttributeDetector("hasLongSleeves", 0.5347));
    _attributeDetectors.push_back(new AttributeDetector("hasShorts", 0.7265));
    _attributeDetectors.push_back(new AttributeDetector("hasJeans", 0.5760));
    _attributeDetectors.push_back(new AttributeDetector("hasLongPants", 0.6399));
    BOOST_FOREACH(AttributeDetector* d, _attributeDetectors)
      d->load();
  }
  //assert(_personDetector.load(std::string(Util::env("HOG_MODELS")))==true);
}

int Identifier::registerPerson(cv::Mat& image,int id) {
  cv::Rect box(0,0,image.cols,image.rows);
  return registerPerson(image, box, id);
}

int Identifier::registerPerson(cv::Mat& image, cv::Rect box, int id) {
  Parts personSegments = _partDetector.getAllSegments(image,box);
  std::vector<bool> attributes = getAttributes(image, box, personSegments);
  std::vector<ColorSignature> signatures;
  signatures.push_back(ColorSignature(image,box));
  cv::Mat face;
  cv::Mat profileFace;
  cv::Mat roiImage = image(box);
  bool ff = false;
  bool pf = true;
  if(personSegments.seen[Parts::Face])
  {
        face = roiImage(personSegments.locations[Parts::Face]).clone();
        ff=true;
  }
  if(personSegments.seen[Parts::FaceProfile])
  {
        profileFace = roiImage(personSegments.locations[Parts::FaceProfile]).clone();
        pf=true;
  }

  //BOOST_FOREACH(bool att,attributes)
  //{
      //printf("%d\t",att);
  //}
  //printf("\n--------------------------%d-----------------------------\n",id);

  bool flag = true;
  BOOST_FOREACH(Identity& i,_identities)
  {
      if(i.id==id)
      {
          i.attributes.push_back(attributes);
          i.signatures.push_back(signatures);
          if(ff)
              i.faces.push_back(face);
          if(pf)
              i.faces.push_back(profileFace);
          flag = false;
      }
  }
//  i.id = __id++;
  if(flag)
  {
    Identity i;
    if(id < 0)
      i.id = id = ++__id;
    else
      i.id = id;
    i.attributes.push_back(attributes);
    i.signatures.push_back(signatures);
    if(ff)
          i.faces.push_back(face);
    if(pf)
          i.faces.push_back(profileFace);
    _identities.push_back(i);
    id = __id;
  }
  return id;
}

bool Identifier::updateFaceRecognizer()
{
  std::vector<cv::Mat> fTrainingImages;
  std::vector<int> trainLabels;
  BOOST_FOREACH(Identity& i,_identities)
  {
      std::vector<cv::Mat>pFaces = i.faces;
      BOOST_FOREACH(cv::Mat face,pFaces)
      {
          if((face.cols*face.rows)>0)
          {
              fTrainingImages.push_back(face);
              trainLabels.push_back(i.id);
          }
      }
  }
  if(fTrainingImages.size()>0)
  {
      //BOOST_FOREACH(cv::Mat im, fTrainingImages)
      //{
          //cv::imwrite("testFace.png",im);
      //}

      _partDetector.TrainFaces(fTrainingImages,trainLabels);
      if(_useProbWeighting==true)
          computeCorrelationAll();
      return true;
  }
  if(_useProbWeighting==true)
      computeCorrelationAll();
  return false;
}

void Identifier::computeCorrelationAll()
{
    BOOST_FOREACH(Identity& i,_identities)
    {
        int m = i.id;
        int n = i.attributes.size();
        PidMat iDataMatrix;
        PidMat iCorrMatrix;
        PidMat iMean;
        PidMat wtMat = cv::Mat::zeros(1,_attributeDetectors.size(),CV_32FC1);
        for(int i=0;i<_attributeDetectors.size();i++)
        {
            wtMat(0,i)=_attributeDetectors[i]->getWeight();
        }
        cv::Mat temp = wtMat.clone();
        for(int i=1;i<_attributeDetectors.size();i++)
        {
            cv::vconcat(wtMat,temp,wtMat);
        }
        bool first = true;
        BOOST_FOREACH(std::vector<bool>currAtt,i.attributes)
        {
            PidMat vector = cv::Mat::zeros(1,_attributeDetectors.size(),CV_32FC1);
            int count = 0;
            BOOST_FOREACH(bool val,currAtt)
            {
                vector(0,count)=(val?1.0f:0.0f);
                count++;
            }
            if(first==true)
            {
                iDataMatrix = vector.clone();
                first=false;
            }
            else
                cv::vconcat(iDataMatrix,vector,iDataMatrix);
        }

//        iDataMatrix = cv::Mat::ones(_attributeDetectors.size(),_attributeDetectors.size(),CV_32FC1);

//        for(int i =0;i<iDataMatrix.rows;i++)
//        {
//            for(int j = 0;j<iDataMatrix.cols;j++)
//                printf("%2.2f:\t",iDataMatrix(i,j));
//            printf("\n");
//        }

//        cv::calcCovarMatrix(iDataMatrix,iCorrMatrix,iMean,0|CV_COVAR_NORMAL|CV_COVAR_ROWS,CV_32FC1);
//        for(int i =0;i<iCorrMatrix.rows;i++)
//        {
//            for(int j = 0;j<iCorrMatrix.cols;j++)
//                printf("%2.2f:\t",iCorrMatrix(i,j));
//            printf("\n");
//        }
//        for(int i =0;i<iMean.rows;i++)
//        {
//            for(int j = 0;j<iMean.cols;j++)
//                printf("%2.2f:\t",iMean(i,j));
//            printf("\n");
//        }
        iCorrMatrix = cv::Mat::eye(_attributeDetectors.size(),_attributeDetectors.size(),CV_32FC1); //hack
        iMean = cv::Mat::zeros(1,_attributeDetectors.size(),CV_32FC1); //hack
        iCorrMatrix = iCorrMatrix.mul(wtMat);
        corrMatrices.push_back(std::make_pair(std::make_pair(iCorrMatrix,iMean),i.id));
    }
}

int Identifier::recognizeFace(const cv::Mat& image, cv::Rect box) {
  cv::Mat face = image(box);
  return _partDetector.recogFace(face);
}

std::vector<bool> Identifier::getAttributes(cv::Mat& image, cv::Rect box, Parts& p) {
  std::vector<bool> attributes;
  BOOST_FOREACH(AttributeDetector* detector, _attributeDetectors) {
    Parts::Location attributeLoc = detector->getDetectLoc();
    cv::Rect loc;
    if(p.seen[attributeLoc])
      loc = p.locations[attributeLoc];
    else
      loc = box;
    //hack -> no part based attribute detector yet
    loc = box;
    bool success = detector->hasAttribute(image, box);
    attributes.push_back(success);
  }
  return attributes;
}

std::vector<Identity> Identifier::attributeMatches(std::vector<bool> attributes) {
  std::vector<Identity> identities;
  //double maxDistance = 1.5; // decrease this when we're ready
  double maxDistance = 1000;
  BOOST_FOREACH(Identity& identity, _identities) {
    double distance = 0;
    std::vector<bool> firstAttribute = identity.attributes[0]; //Ideally we should prababilistically determine Identity from attribute vectors
    for(int i = 0; i < attributes.size(); i++)
      if(attributes[i] != firstAttribute[i])
        distance += _attributeDetectors[i]->getWeight();
    if(distance < maxDistance)
      identities.push_back(identity);
    //printf("distance to %i: %2.2f\n", identity.id, distance);
  }
  return identities;
}

std::vector<Identity> Identifier::probabilisticMatches(std::vector<bool> attributes)
{
    //We can also just return the first 5 or something because allMatches is sorted
    std::vector<Identity> identities;
    double thresh = 1; //need to figure out this value
    std::vector<std::pair<double,int> > allMatches = likelyPersonID(attributes);
    std::vector<int> probableIds;
    std::pair<double,int> pairProbId;
    BOOST_FOREACH(pairProbId,allMatches)
    {
        if(pairProbId.first>thresh)
            probableIds.push_back(pairProbId.second);
    }

    BOOST_FOREACH(Identity& identity, _identities)
    {
        BOOST_FOREACH(int id,probableIds)
        {
            if(id==identity.id)
                identities.push_back(identity);
        }
    }
    return identities;
}

std::vector<std::pair<double,int> > Identifier::likelyPersonID(std::vector<bool> attributes)
{
    PidMat vector = cv::Mat::zeros(1,_attributeDetectors.size(),CV_32FC1);
    int i = 0;
    float classProb = float(((float)1/(float)__id));
    std::vector<std::pair<double,int> > estProbabilities;
//    double max = 0;
//    int retId = -1;
    BOOST_FOREACH(bool val,attributes)
    {
        vector(0,i) = (val?1.0f:0.0f);
        i++;
    }
    std::pair<std::pair<PidMat,PidMat>,int> datMat;
    BOOST_FOREACH(datMat,corrMatrices)
    {
        std::pair<PidMat,PidMat> corrMatMean = datMat.first;
        PidMat corrMat = corrMatMean.first;
        PidMat meanMat = corrMatMean.second;
        PidMat corrMatinv = corrMat.inv();
        PidMat inpMat = vector-meanMat;
        double dist = cv::Mahalanobis(inpMat,inpMat,corrMatinv);
        double det = abs(cv::determinant(corrMat));
        double currProb = -(dist/2)+log(classProb)-(det/2);
//        if(currProb>max)
//        {
//            max=currProb;
//            retId = datMat.second;
//        }
        estProbabilities.push_back(std::make_pair(currProb,datMat.second));
    }
    std::sort(estProbabilities.begin(),estProbabilities.end(),Identifier::pairCompare);
    return estProbabilities;
}

bool Identifier::pairCompare(const std::pair<double,int>& x,const std::pair<double,int>& y)
{
    if(x>y)
        return true;
    return false;
}

int Identifier::identifySignature(ColorSignature& signature, std::vector<Identity> identities) {
  double maxDistance = 5000.00;
  int bestID = UNIDENTIFIED_PERSON;
  double bestDistance = maxDistance;
  BOOST_FOREACH(Identity& identity, identities) {
    //printf("id\n");
    std::vector<ColorSignature> signatures = identity.signatures[0]; //Ideally we should prababilistically determine Identity from color signatures
    BOOST_FOREACH(ColorSignature& testSignature, signatures) {
      double distance = signature.distanceTo(testSignature);
      if(distance < maxDistance) {
        if(distance < bestDistance) {
          bestID = identity.id;
          bestDistance = distance;
        }
      }
    }
  }
  return bestID;
}

int Identifier::identifyPerson(cv::Mat& image) {
   cv::Rect box(0,0,image.cols,image.rows);
   return identifyPerson(image, box);
}

int Identifier::identifyPerson(cv::Mat& image, cv::Rect box) {
  //int faceID = recognizeFace(image, box);
  //if(faceID!=UNIDENTIFIED_PERSON) return faceID;
  Parts personSegments = _partDetector.getAllSegments(image,box);
  std::vector<bool> attributes = getAttributes(image, box,personSegments);
  std::vector<Identity> validIdentities;
  if(_useProbWeighting)
    validIdentities = probabilisticMatches(attributes);
  else
    validIdentities = attributeMatches(attributes);
  if(validIdentities.size() > 0) {
    ColorSignature signature(image, box);
    return identifySignature(signature, validIdentities);
  }   
  return UNIDENTIFIED_PERSON;
}
