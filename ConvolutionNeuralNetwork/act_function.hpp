//
//  act_function.h
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 3/4/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef act_function_hpp
#define act_function_hpp

#include <stdio.h>
#include <iostream>
#include <armadillo>
#include <functional>
#include "inl_nn_funcs.hpp"
#include "const_value.hpp"

using namespace std;
using namespace arma;

class ActFunction {
 private:
  string func_name_;
  bool is_hid_;
  function<mat(mat)> custom_func_;
  function<mat(mat)> custom_diff_func_;

 public:
  ActFunction(string func_name, bool is_hid):
    func_name_(func_name),
    is_hid_(is_hid),
    custom_func_(nullptr),
    custom_diff_func_(nullptr) {};
  
  ActFunction(function<mat(mat)> custom_func, function<mat(mat)> custom_diff_func, bool is_hid):
  func_name_(kUserDefinedMethod),
  is_hid_(is_hid),
  custom_func_(custom_func),
  custom_diff_func_(custom_diff_func) {};
  
  mat  ComputeActFunc(const mat &input) const;
  mat  DiffActFunc(const mat &input) const;
  void SetCustomFunc(function<mat(mat)> custom_func, function<mat(mat)> custom_diff_func);
  
  string get_func_name() const;
  void set_func_name(string func_name);
  bool get_is_hid() const;
  
  
};
#endif /* act_function_h */
