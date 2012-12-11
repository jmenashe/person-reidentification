cmake_minimum_required(VERSION 2.8)

SET(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

LINK_DIRECTORIES(/usr/local/lib /usr/lib)
INCLUDE_DIRECTORIES(/opt/X11/include)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(dlib)
INCLUDE_DIRECTORIES(libsvm)
#INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
set(CMAKE_BUILD_TYPE Debug)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost 1.46 COMPONENTS iostreams thread REQUIRED)

ADD_LIBRARY(svm libsvm/libSvm.h include/LibsvmSvm.h include/DlibSvm.h include/OcvSvm.h libsvm/libSvm.cpp src/LibsvmSvm.cpp src/DlibSvm.cpp src/OcvSvm.cpp)
ADD_LIBRARY(util include/Util.h src/Util.cpp)
ADD_LIBRARY(extractor include/FeatureExtractor.h include/SiftFeatureExtractor.h include/HogFeatureExtractor.h include/PHogFeatureExtractor.h src/FeatureExtractor.cpp src/HogFeatureExtractor.cpp src/SiftFeatureExtractor.cpp src/PHogFeatureExtractor.cpp)
ADD_LIBRARY(identifier include/AttributeDetector.h include/PartDetector.h include/PersonDetector.h include/ColorSignature.h include/Identifier.h include/Typedefs.h include/Parts.h include/Histogram.h include/Environment.h include/Identity.h src/AttributeDetector.cpp src/PartDetector.cpp src/PersonDetector.cpp src/ColorSignature.cpp src/Identifier.cpp dlib/dlib/all/source.cpp)

ADD_EXECUTABLE(main src/main.cpp)
TARGET_LINK_LIBRARIES(main util identifier svm extractor ${OpenCV_LIBS} yaml-cpp ${Boost_LIBRARIES})

#ADD_EXECUTABLE(c_analysis src/c_analysis.cpp)
#TARGET_LINK_LIBRARIES(c_analysis util identifier ${OpenCV_LIBS} yaml-cpp ${Boost_LIBRARIES})

ADD_EXECUTABLE(a_trainer src/a_trainer.cpp)
TARGET_LINK_LIBRARIES(a_trainer util identifier svm extractor ${OpenCV_LIBS} yaml-cpp ${Boost_LIBRARIES})

ADD_EXECUTABLE(f_trainer src/f_trainer.cpp)
TARGET_LINK_LIBRARIES(f_trainer util identifier ${OpenCV_LIBS} yaml-cpp ${Boost_LIBRARIES})

ADD_EXECUTABLE(parts_detect src/parts_detect.cpp)
TARGET_LINK_LIBRARIES(parts_detect util identifier ${OpenCV_LIBS} yaml-cpp ${Boost_LIBRARIES})
