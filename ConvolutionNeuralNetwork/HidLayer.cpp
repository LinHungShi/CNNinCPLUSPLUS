//
//  HidLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/22/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "HidLayer.hpp"

bool HidLayer::UpdateDelta(mat next_layer_delta,
                           mat next_layer_weight)
{

    mat der_output = DDiffActFunc(output_, act_func_);
    mat result = der_output % (next_layer_delta * next_layer_weight.t());
    delta_ = result;
    return true;
}

bool HidLayer::UpdateParm(double alpha,
                          mat next_layer_delta,
                          mat next_layer_weight,
                          mat input)
{
    
    UpdateDelta(next_layer_delta, next_layer_weight);
    UpdateWeightGradient(input);
    UpdateWeight(alpha);
    return true;
}

bool HidLayer::UpdateOutput(mat input)
{
    
    mat pre_act = input * weight_;
    output_ = DComputeActFunc(pre_act, act_func_);
    return true;

}