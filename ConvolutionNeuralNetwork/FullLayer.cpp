//
//  FullLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/19/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "FullLayer.hpp"

bool FullLayer::UpdateWeightGradient(mat input)
{
    
    gradient_ = input.t() * delta_ ;
    return true;
    
}


void FullLayer::InitWeight(int inp_dim)
{
    
    //cout << "inp_dim: " << inp_dim << endl;
    //cout << "num_neuron: " << num_neuron_ << endl;
    
    weight_ = (*w_init_func_)(inp_dim, num_neuron_);
    
}