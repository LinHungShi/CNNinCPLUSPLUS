//
//  default_weight_init_funcs.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/23/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef default_weight_init_funcs_hpp
#define default_weight_init_funcs_hpp

#include <stdio.h>
#include <armadillo>
#include <iostream>
#include <map>
using namespace std;
using namespace arma;

mat InitWeight(int row, int col, string init_method_name);

#endif /* default_weight_init_funcs_hpp */
