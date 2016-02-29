//
//  hidden_act_functions.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/21/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "hidden_act_function.hpp"

mat  HidActFunction::ComputeActFunc(const mat input){
    
    if (act_func_name_ == "self-made")
    {
        
        return custom_act_func(input);
    
    }
    
    return DComputeActFunc(input, act_func_name_);
    
}

mat  HidActFunction::DiffActFunc(const mat input){
    
    if (act_func_name_ == "self-made")
    {
        
        return custom_diff_act_func(input);
        
    }
    
    return DDiffActFunc(input, act_func_name_);
    
}




