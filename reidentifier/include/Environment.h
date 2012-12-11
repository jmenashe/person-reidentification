#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <assert.h>
std::string HOME_DIR = getenv("HOME");
std::string SVM_MODELS = getenv("SVM_MODELS");
std::string HOG_MODELS = getenv("HOG_MODELS");
std::string HAAR_MODELS = getenv("HAAR_MODELS");
std::string FACE_MODELS = getenv("FACE_MODELS");
std::string IM_SAVE_PATH = getenv("IM_SAVE_PATH");
assert(HOME_DIR != "");
assert(SVM_MODELS != "");
assert(HOG_MODELS != "");
assert(HAAR_MODELS != "");
assert(FACE_MODELS != "");
assert(IM_SAVE_PATH != "");

#endif
