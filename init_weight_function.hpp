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
#include "default_weight_init_funcs.hpp"
using namespace std;
using namespace arma;


class InitWeightFunction{

public:
    
    string init_method_name_;
    function<mat(int, int)> custom_init_method_;
    
    InitWeightFunction(string init_method_name):init_method_name_(init_method_name){};
    
    InitWeightFunction(function<mat(int, int)> func):custom_init_method_(func),
        init_method_name_("self-made"){};
    
    mat operator()(int row, int col);
    
};
#endif /* init_weight_function_hpp */
