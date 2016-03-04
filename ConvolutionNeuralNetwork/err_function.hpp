//
//  err_function.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef err_function_hpp
#define err_function_hpp

#include <stdio.h>
#include <iostream>
#include <armadillo>
#include <functional>
#include "inl_nn_funcs.hpp"
#include "const_value.hpp"
using namespace std;
using namespace arma;
class ActFunction;



class ErrFunction{
 private:
  string func_name_;
  function<double(mat const&)> custom_func_;
  function<mat(mat const&, mat const&, ActFunction const&)> custom_diff_func_;
  
public:
  // Creators
  
  // Use the default function to create error function. Name of default_func_name must be
  // one of supported error function.
  ErrFunction(string default_func_name):
    func_name_(default_func_name),
    custom_func_(nullptr),
    custom_diff_func_(nullptr) {};
  
  // Use the customized function to construct error function. For differentiation, user must supply the output activation function
  ErrFunction(function<double(mat const&)>default_err_func_,
              function<mat(mat const&, mat const&, ActFunction const&)> default_diff_func_):
    custom_func_(default_err_func_),
    custom_diff_func_(default_diff_func_),
    func_name_(kUserDefinedMethod) {};
    
  // Manipulators
  double ComputeErrFunc(mat const&, mat const&) const;
  mat DiffErrFunc(mat const&, mat const&, ActFunction const &output_func) const;
  
  //Accessors and Setters
  string get_func_name() { return func_name_; }
  
  // Set error function to the default function
  // Set custom_func_ and tis derivative to nullptr to keep the consistency in class
  void set_func_name(string func_name) {
    func_name_ = func_name;
    custom_func_ = nullptr;
    custom_diff_func_ = nullptr;
  }
  
  function<double(mat const&)> get_custom_func(){ return custom_func_; }
  function<mat(mat const&, mat const&, ActFunction const&)> get_custom_diff_func(){ return custom_diff_func_; }
  
  // User must customize the error function and its derivative together in order to select proper function.
  // Set the custom_func_ to user defined function
  void SetCustomErrFunc(function<double(mat const&)> custom_func,
                   function<mat(mat const&, mat const&, ActFunction const&)> custom_diff_func) {

    custom_func_ = custom_func;
    custom_diff_func_ = custom_diff_func;
    set_func_name(kUserDefinedMethod);
  }
};
#endif /* err_function_hpp */
