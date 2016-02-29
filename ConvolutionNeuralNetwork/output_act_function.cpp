//
//  output_act_function.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "output_act_function.hpp"


mat  OutputFunction::ComputeActFunc(mat const input){
    
    if (act_func_name_ == "self-made")
    {
        
        return custom_act_func(input);
        
    }
    
    return DComputeOutputFunc(input, act_func_name_);
    
}

mat  OutputFunction::DiffActFunc(mat const input){
    
    if (act_func_name_ == "self-made")
    {
        
        return custom_diff_act_func(input);
        
    }
    
    return DDiffOutputFunc(input, act_func_name_);
    
}
