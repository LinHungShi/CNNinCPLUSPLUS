//
//  init_weight_function.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/23/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "init_weight_function.hpp"



mat InitWeightFunction::operator()(int row, int col) {
  if (method_name_ == kUserDefinedMethod) {
      return custom_method_(row, col);
  }
  return InitWeight(row, col, method_name_);
}