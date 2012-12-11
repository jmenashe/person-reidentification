#include "DlibSvm.h"

DlibSvm::DlibSvm() {
}

void DlibSvm::train(const std::vector<PidMat>& features,const std::vector<int>& labels) {
  assert(features.size() > 0 && labels.size() == features.size());

  std::vector<Sample> samples;
  std::vector<double> dlabels;
  dlib::svm_c_linear_trainer<Kernel> trainer;
  dlib::vector_normalizer<Sample> normalizer;

  for(int i = 0; i < features.size(); i++) {
    PidMat feature = features[i];
    Sample s;
    //printf("%i (%i): ", i, labels[i]);
    s.set_size(feature.rows, 1);
    for(int j = 0; j < feature.rows; j++) {
      s(j) = feature(j,0);
      //printf("%2.2f ", s(j));
    }
    //printf("\n");
    samples.push_back(s);
    dlabels.push_back(labels[i]);
  }

  //normalizer.train(samples);
  //for(int i = 0; i < samples.size(); i++)
  //samples[i] = normalizer(samples[i]);

  const double max_nu = dlib::maximum_nu(dlabels);
  printf("max nu: %2.4f\n", max_nu);

  //double sum = 0;
  //for(double gamma = 0.01; gamma <= 1; gamma += .01) {
    //for(double nu = 0.0001; nu < .0002; nu += .000005) {
      //trainer.set_kernel(Kernel(gamma));
      //trainer.set_nu(nu);
      //dlib::matrix<double, 1, 2> result = dlib::cross_validate_trainer(trainer, samples, dlabels, 3);
      //double score = result(0) + result(1) - fabs(result(0) - result(1));
      //if(score > sum) {
        //sum = score;
        //printf("gamma: %2.5f, nu: %2.6f\n", gamma, nu);
        //std::cout << "x-validate: " << result << "\n";
      //}
    //}
  //}
  //trainer.set_kernel(Kernel(0.5, 5, 3));
  for(int c = 100; c <= 10000000; c *= 10) {
    for(double eps = .001; eps <= 2; eps += .01) {
      trainer.set_c(c);
      trainer.set_epsilon(eps);
      dlib::matrix<double, 1, 2> result = dlib::cross_validate_trainer(trainer, samples, dlabels, 3);
      printf("eps: %2.2f, c: %i, TPR: %2.4f, TNR: %2.4f\n", eps, c, result(0), result(1));
    }
  }

  dlib::randomize_samples(samples, dlabels);
  trainer.set_c(10);
  trainer.set_epsilon(.001);

  //_trainer.set_kernel(Kernel(0.15625));
  //_trainer.set_nu(0.15625);
  //_svm.normalizer = normalizer;
  //_svm = dlib::train_probabilistic_decision_function(trainer, samples, dlabels, 3);
  _svm = trainer.train(samples, dlabels);
}

float DlibSvm::predict(const PidMat& feature) {
  Sample s;
  s.set_size(feature.rows, 1);
  for(int j = 0; j < feature.rows; j++)
    s(j) = feature(j,0);

  float result = _svm(s);
  return result;
}

void Svm::save(std::string file) {
}

void Svm::load(std::string file) {
}

