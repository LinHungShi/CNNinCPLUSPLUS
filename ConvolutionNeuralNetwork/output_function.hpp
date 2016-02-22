//
//  output_function.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef output_function_hpp
#define output_function_hpp

#include <stdio.h>
#include <iostream>
#include <armadillo>
#include <functional>
#include "act_function.hpp"

using namespace std;
using namespace arma;

class OutputFunction{
    
public:
    
    string output_func_name_;
    string diff_output_func_name_;
    function<mat(mat)> custom_output_func;
    function<mat(mat)> custom_diff_output_func;
    
    OutputFunction(string default_output_func_name,
                string default_diff_out_func_name):output_func_name_(default_output_func_name),
                                                   diff_output_func_name_(default_output_func_name){};
    
    OutputFunction(function<mat(mat)> output_func,
                   function<mat(mat)> diff_output_func):custom_output_func(output_func),
                                                        custom_diff_output_func(diff_output_func),
                                                        output_func_name_("self-made"),
                                                        diff_output_func_name_("self-made"){};
    
    mat ComputeOutputFunc(mat const);
    mat DiffOutputFunc(mat const);

    
};
#endif /* output_function_hpp */
