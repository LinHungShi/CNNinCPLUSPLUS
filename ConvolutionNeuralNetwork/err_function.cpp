//
//  err_function.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "err_function.hpp"



double  ErrFunction::ComputeErrFunc(const mat pred, mat const y){
    
    if (err_func_name_ == "self-made")
    {
        
        return custom_err_func(pred);
        
    }
    
    return DComputeErrFunc(y, pred, err_func_name_);
    
}


mat  ErrFunction::DiffErrFunc(const mat pred,
                              mat const y,
                              OutputFunction* output_func){
    
    if (err_func_name_ == "self-made")
    {
        
        return custom_diff_err_func(pred, y, output_func);
        
    }
    
    return DDiffErrFunc(pred, y, err_func_name_);
    
}