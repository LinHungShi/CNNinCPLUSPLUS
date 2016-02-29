//
//  output_act_function.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef output_act_function_hpp
#define output_act_function_hpp

#include <stdio.h>
#include <iostream>
#include <armadillo>
#include <functional>
#include "act_function.hpp"
#include "ActFuncOp.hpp"

using namespace std;
using namespace arma;

class OutputFunction:public ActFunction{
    
public:
    
    OutputFunction(string default_act_func_name):ActFunction(default_act_func_name){};
    OutputFunction(function<mat(mat)> act_func,
                   function<mat(mat)> diff_act_func):ActFunction(act_func,
                                                                 diff_act_func){};
    
    mat ComputeActFunc(mat const);
    mat DiffActFunc(mat const);

    
};
#endif /* output_function_hpp */
