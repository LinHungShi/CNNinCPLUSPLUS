//
//  act_function.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "act_function.hpp"


mat  ActFunction::ComputeActFunc(const mat &input) const{
  if (func_name_ == kUserDefinedMethod) {
    
    return custom_func_(input);
    
  }
  if (is_hid_) return DComputeActFunc(input, func_name_);
  else return DComputeOutputFunc(input, func_name_);
  
}

mat  ActFunction::DiffActFunc(const mat &input) const{
  if (func_name_ == kUserDefinedMethod) {
    return custom_diff_func_(input);
}
  if(is_hid_) return DDiffActFunc(input, func_name_);
  else return DDiffOutputFunc(input, func_name_);
}

void ActFunction::SetCustomFunc(function<mat (mat)> custom_func, function<mat (mat)> custom_diff_func) {
  
  custom_func_ = custom_func;
  custom_diff_func_ = custom_diff_func;
  func_name_ = kUserDefinedMethod;
}

string ActFunction::get_func_name() const {
  return func_name_;
}

void ActFunction::set_func_name(string func_name) {
  func_name_ = func_name;
  custom_diff_func_ = nullptr;
  custom_func_ = nullptr;
}

bool ActFunction::get_is_hid() const{
  return is_hid_;
}