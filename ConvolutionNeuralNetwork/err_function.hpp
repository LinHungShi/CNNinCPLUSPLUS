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
#include "ActFuncOp.hpp"
#include "output_function.hpp"

using namespace std;
using namespace arma;

class ErrFunction{
    
public:

    string err_func_name_;
    string diff_err_func_name_;
    
    function<double(mat)> custom_err_func;
    function<mat(mat, OutputFunction*)> custom_diff_err_func;
    
    ErrFunction(string default_err_func_name,
                string default_diff_err_func_name):err_func_name_(default_err_func_name),
                                                   diff_err_func_name_(default_diff_err_func_name){};
    
    ErrFunction(function<double(mat)>default_err_func_,
                function<mat(mat, OutputFunction*)>default_diff_err_func_):custom_err_func(default_err_func_),
                                                          custom_diff_err_func(default_diff_err_func_){};
    
    double ComputeErrFunc(mat const, mat const);
    mat DiffErrFunc(mat const, mat const, OutputFunction *output_func);

};
#endif /* err_function_hpp */
