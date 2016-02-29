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

    mat der_output = act_func_.DiffActFunc(output_);
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
    output_ = act_func_.ComputeActFunc(pre_act);
    return true;

}
ostream &operator<<(ostream &stream, HidLayer &layer)
{
    int width = 5;
    int prec = 4;
    stream << "Layer name: " << setw(width) << layer.name_ << endl;
    stream << "Number of Neuron:" << setw(width) << layer.num_neuron_ << endl;
    stream << "Activation Function: " << setw(width) << layer.act_func_.act_func_name_ << endl;
    
    if(layer.has_w_init_func_)
    {
        
        stream << "Weight Initialization: " << setw(width) << layer.w_init_func_->init_method_name_;
    
    }
    else
    {
        
        stream << "Weight Initialization: " << setw(width) << "None"<< endl;
    
    }
    
    stream << setprecision(prec) << "Weight Dimension: " << setw(width) << "(" << layer.weight_.n_rows << "," << layer.weight_.n_cols << ")" << endl;
    
    stream << PrintSepLine()<< endl;
    return stream;
    
}