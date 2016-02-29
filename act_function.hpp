//
//  act_function.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef act_function_hpp
#define act_function_hpp

#include <stdio.h>
#include <iostream>
#include <functional>
#include <armadillo>
using namespace arma;
using namespace std;

class ActFunction{
    
public:
    
    string act_func_name_;
    
    function<mat(mat)> custom_act_func;
    function<mat(mat)> custom_diff_act_func;
    
    ActFunction(string default_act_func_name):act_func_name_(default_act_func_name){};
    
    ActFunction(function<mat(mat)> act_func,
                function<mat(mat)> diff_act_func):custom_act_func(act_func),
                                                  custom_diff_act_func(diff_act_func),
                                                  act_func_name_("self-made"){};
    
    virtual mat ComputeActFunc(mat const)=0;
    virtual mat DiffActFunc(mat const)=0;
    
    
};

#endif /* act_function_hpp */
