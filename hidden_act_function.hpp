//
//  hidden_act_functions.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/21/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef hidden_act_function_hpp
#define hidden_act_function_hpp

#include <stdio.h>
#include <iostream>
#include <functional>
#include <armadillo>
#include "ActFuncOp.hpp"
#include "act_function.hpp"
using namespace std;
using namespace arma;

class HidActFunction:public ActFunction{
    
public:
    
    HidActFunction(string default_act_func_name):ActFunction(default_act_func_name){};
    HidActFunction(function<mat(mat)> act_func,
                function<mat(mat)> diff_act_func):ActFunction(act_func,
                                                  diff_act_func){};
    
    mat ComputeActFunc(mat const);
    mat DiffActFunc(mat const);
    
    
};
#endif /* nn_functions_hpp */
