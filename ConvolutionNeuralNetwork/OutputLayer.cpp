//
//  OutputLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "OutputLayer.hpp"


//bool OutputLayer::updatePar(double alpha, Layer const *n_layer, Layer const *p_layer){
    
    
//    cout << "wrong function" << endl;
//    return true;
//}

bool OutputLayer::UpdateParm(double alpha,
                             mat y,
                             mat input,
                             ErrFunction &err_func)
{

    UpdateDelta(y, err_func);
    UpdateWeightGradient(input);
    UpdateWeight(alpha);


    return true;
}

bool OutputLayer::UpdateDelta(mat y, ErrFunction &err_func){
    
    
    delta_ = err_func.DiffErrFunc(output_, y, &output_func_);
    return true;
}

bool OutputLayer::UpdateOutput(mat input){
    
    mat pre_act = input * weight_;
    output_ = output_func_.ComputeActFunc(pre_act);
    return true;
}

ostream &operator<<(ostream &stream, OutputLayer &layer)
{
    int width = 5;
    int prec = 4;
    cout << layer.name_;
    stream << "Layer name: " << setw(width) << layer.name_ << endl;
    stream << "Number of Neuron:" << setw(width) << layer.num_neuron_ << endl;
    stream << "Output Function: " << setw(width) << layer.output_func_.act_func_name_ << endl;
    
    if(layer.has_w_init_func_)
    {
        
        stream << "Weight Initialization: " << setw(width) << layer.w_init_func_->init_method_name_;
        
    }
    else
    {
        
        stream << "Weight Initialization: " << setw(width) << "None" << endl;;
        
    }
    
    stream << setprecision(prec) << "Weight Dimension: " << setw(width) << "(" << layer.weight_.n_rows << "," << layer.weight_.n_cols << ")" << endl;
    
    stream << PrintSepLine()<< endl;
    return stream;
    
}