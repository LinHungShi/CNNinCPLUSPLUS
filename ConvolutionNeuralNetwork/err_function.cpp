//
//  err_function.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "err_function.hpp"



double  ErrFunction::ComputeErrFunc(mat const &pred, mat const &y) const {
  if (func_name_ == kUserDefinedMethod)
    {
        return custom_func_(pred);
    }
  return DComputeErrFunc(y, pred, func_name_);
}


mat  ErrFunction::DiffErrFunc(mat const &pred, mat const &y, ActFunction const &output_func) const {
  if (func_name_ == kUserDefinedMethod)
    {
        return custom_diff_func_(pred, y, output_func);
    }
  return DDiffErrFunc(pred, y, func_name_, output_func);
}