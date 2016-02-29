//
//  Layer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <stdio.h>
#include <armadillo>

using namespace arma;
using namespace std;


class Layer{
 public:
  string name_;
  int num_neuron_;
  mat output_, delta_;
    Layer(int num_neuron, string name):num_neuron_(num_neuron), name_(name){};
 
};

#endif /* Layer_hpp */