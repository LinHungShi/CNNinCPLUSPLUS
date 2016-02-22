//
//  Layer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//


#include "Layer.hpp"


Layer::Layer(int input_size,
             int num_neuron,
             string init_method,
             string act_func,
             string var_name)
{
    
    num_neuron_ = num_neuron;
    InitWeight(input_size, num_neuron, init_method);
    act_func_ = act_func;
    name_ = var_name;

}


void Layer::InitWeight(int row, int col, string init_method)
{
    
    if(init_method == "randn")
    {
        
        this -> weight_ = mat(row, col, fill::randn);
    
    }
    
}

