#include "PHogFeatureExtractor.h"

PHogFeatureExtractor::PHogFeatureExtractor() {
}

PidMat PHogFeatureExtractor::extractFeature(const cv::Mat& image) {

  assert(image.channels() == 1);

  bool display = false;
//    cv::namedWindow("Edge image");
//    cv::namedWindow("Gradients X");
//    cv::namedWindow("Gradients Y");
//    cv::namedWindow("Gradient Magnitudes");
//    cv::namedWindow("Gradient Angles");
//    cv::namedWindow("Bin Matrix");
//    cv::namedWindow("descriptor");

  int m_nLevels = 3;
  int m_nBins = 20;
  double m_cannyLowThreshold = 85.0;
  double m_cannyHighThreshold = 110.0;
  int m_maxAngle = 360;
  // convert image to gray scale
//    cv::Mat gray_8U(image.size(), CV_8U);
//    cv::cvtColor(image, gray_8U, CV_BGR2GRAY);
  cv::Mat gray_8U = image.clone();
  // perform canny edge detection
  cv::Mat edges_8U(gray_8U.size(), CV_8U);
  cv::Canny(gray_8U, edges_8U, m_cannyLowThreshold, m_cannyHighThreshold);
  if (display)
  {
      cv::imshow("Edge image", edges_8U);
  }
  // calculate gradients
  cv::Mat gradientX_32F(gray_8U.size(), CV_32F);
  cv::Mat gradientY_32F(gray_8U.size(), CV_32F);
  cv::Sobel(gray_8U, gradientX_32F, gradientX_32F.type(), 1, 0, 3);
  cv::Sobel(gray_8U, gradientY_32F, gradientY_32F.type(), 0, 1, 3);
  if (display)
  {
      cv::imshow("Gradients X", gradientX_32F/255.0);
      cv::imshow("Gradients Y", gradientY_32F/255.0);
  }
  // calculate gradient angles and magnitudes
  cv::Mat magnitude_32F(gray_8U.size(), CV_32F, cv::Scalar(0.0));
  cv::Mat angles_32F(gradientX_32F.size(),CV_32F);
  bool anglesInDegrees = true;
  cv::cartToPolar(gradientX_32F, gradientY_32F, magnitude_32F, angles_32F, anglesInDegrees);
  /*double minval, maxval;
  cv::minMaxLoc(gradientXY_32F, &minval, &maxval);
  std::cout << minval << std::endl;
  std::cout << maxval << std::endl;
  cv::minMaxLoc(gradientXY, &minval, &maxval);
  std::cout << minval << std::endl;
  std::cout << maxval << std::endl;*/
  if (display)
  {
      cv::imshow("Gradient Magnitudes", magnitude_32F/255.0);
      cv::imshow("Gradient Angles", angles_32F/(360));
  }
  // create matrix containing histogram values for each pixel
  cv::Mat edgeAngles_32F(angles_32F.size(), CV_32F, cv::Scalar(-1));
  angles_32F.copyTo(edgeAngles_32F, edges_8U);
  float degreesPerBin = static_cast<float>(m_maxAngle) / m_nBins;
  cv::Mat bins_32F = (edgeAngles_32F / degreesPerBin);
  // convert to int
  cv::Mat bins_32S(angles_32F.size(), CV_32S);
  bins_32F -= cv::Mat(edgeAngles_32F.size(), CV_32F, cv::Scalar(0.5));
  bins_32F.convertTo(bins_32S, CV_32S);
  if (display)
  {
      cv::imshow("Bin Matrix", bins_32S);
  }
  // create histograms for all pyramid levels
  int nCells = 0;
  for (int l = 0; l < m_nLevels; ++l)
  {
      nCells += (int)pow((float)4, (float)l);
  }
  cv::Mat descriptor(m_nBins*nCells, 1, CV_32F, cv::Scalar(0.0));
  int bin;
  int cellNo = 0;
  for (int level = 0; level < m_nLevels; ++level)
  {
      int cellWidth = (int)(bins_32S.cols / pow((float)2, (float)level));
      int cellHeight = (int)(bins_32S.rows / pow((float)2, (float)level));
      int cellLeft = 0;
      while (cellLeft + cellWidth <= bins_32S.cols)
      {
          int cellTop = 0;
          while (cellTop + cellHeight <= bins_32S.rows)
          {
              for (int i = cellTop; i < cellTop+cellHeight; ++i)
              {
                  for (int j = cellLeft; j < cellLeft+cellWidth; ++j)
                  {
                      bin = bins_32S.at<int>(i, j);
                      if (bin > 0)
                      {
                          descriptor.at<float>(bin+cellNo*m_nBins, 0) += magnitude_32F.at<float>(i, j);
                      }
                  }
              }
              cellTop += cellHeight;
              cellNo++;
          }
          cellLeft += cellWidth;
      }
  }
  if (display)
  {
      cv::Mat descriptorResized;
      cv::resize(descriptor, descriptorResized, cv::Size(), 3, 50, cv::INTER_NEAREST);
      cv::imshow("descriptor", descriptorResized / cv::norm(descriptorResized, cv::NORM_INF));
      cv::waitKey(10);
  }
  assert(descriptor.cols == 1);
  assert(descriptor.rows > 0);
  return descriptor;
}

