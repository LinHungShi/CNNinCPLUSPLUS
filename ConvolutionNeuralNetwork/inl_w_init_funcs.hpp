//
//  default_weight_init_funcs.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/23/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef inl_w_init_func_hpp
#define inl_w_init_func_hpp

#include <stdio.h>
#include <armadillo>
#include <iostream>
#include <map>
using namespace std;
using namespace arma;
typedef mat(*WeightInit)(int row, int col);


inline mat Randn(int row, int col) {
  return randn<mat>(row, col);
}

inline mat Randu(int row, int col) {
  return randu<mat>(row,col);
}

inline mat Ones(int row, int col) {
  return ones(row, col);
}

mat InitWeight(int row, int col, string init_method_name);



#endif /* default_weight_init_funcs_hpp */
