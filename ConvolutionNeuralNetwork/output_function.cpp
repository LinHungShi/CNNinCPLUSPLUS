//
//  output_function.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/22/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "output_function.hpp"


mat  OutputFunction::ComputeOutputFunc(const mat input){
    
    if (output_func_name_ == "self-made")
    {
        
        return custom_output_func(input);
        
    }
    
    return DComputeOutputFunc(input, output_func_name_);
    
}

mat  OutputFunction::DiffOutputFunc(const mat input){
    
    if (diff_output_func_name_ == "self-made")
    {
        
        return custom_diff_output_func(input);
        
    }
    
    return DDiffOutputFunc(input, output_func_name_);
    
}
