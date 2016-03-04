//
//  init_weight_function.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/23/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef init_weight_function_hpp
#define init_weight_function_hpp

#include <stdio.h>
#include <functional>
#include <armadillo>
#include <iostream>
#include "inl_w_init_funcs.hpp"
#include "const_value.hpp"
using namespace std;
using namespace arma;


class InitWeightFunction{
 private:
  string method_name_;
  function<mat(int, int)> custom_method_;
 
 public:
  // Constructors
  // User use default method to initialize weight
  InitWeightFunction(string init_method_name):
    method_name_(init_method_name),
    custom_method_(nullptr) {};
  
  //User use customized function to initialize weight
  InitWeightFunction(function<mat(int, int)> func):
    custom_method_(func),
    method_name_(kUserDefinedMethod){};
  
  // Manipulators
  mat operator()(int row, int col);
  
  // Accessors and Setters
  inline string get_method_name() { return method_name_; }
  inline void set_method_name(string name) {
    method_name_ = name;
    custom_method_ = nullptr;
  }
  inline function<mat(int, int)> get_custom_method() { return custom_method_; }
  inline void set_custom_method(function<mat(int, int)> custom_method) {
    method_name_ = kUserDefinedMethod;
    custom_method_ = custom_method;
  }
};
#endif /* init_weight_function_hpp */
