//
//  Layer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef base_layer_hpp
#define base_layer_hpp

#include <stdio.h>
#include <armadillo>

class ActFunction;

using namespace arma;
using namespace std;

class BaseLayer {
 protected:
  const string c_name_;
  int num_neuron_;
  BaseLayer *prev_layer_;
  BaseLayer *next_layer_;
  //delta_ refers to the derivative of error function with respect to pre activation function
  mat output_, delta_;
  
  //Users are allowed to customize activation function or use the default functions
  
 public:
  // Constructor
  BaseLayer(int num_neuron, string name):num_neuron_(num_neuron), c_name_(name){};
  
  // BaseLayer is a virtual class, we need to create destructor
  virtual ~BaseLayer(){};
  
  // Manipulator
  virtual void updateParam() = 0;
  //Acessors and Setter
  //Provide only accessors for name_ and num_neuron
  string get_name() const{ return c_name_; }
  int get_num_neuron() const { return num_neuron_; }
  
  //Provide accessors and setters for output_, delta_ and act_func_
  mat get_output() const { return output_; }
  void set_output(mat &output) { output_ = output; }
  mat get_delta() const { return delta_; }
  void set_delta(mat &delta) { delta_ = delta; }

};

#endif /* Layer_hpp */