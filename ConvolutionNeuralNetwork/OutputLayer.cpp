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
                             string err_func)
{

    UpdateDelta(y, err_func);
    UpdateWeightGradient(input);
    UpdateWeight(alpha);
    return true;
}

bool OutputLayer::UpdateDelta(mat y, string err_func){
    
    delta_ = DDiffErrFunc(output_, y, err_func);
    return true;
}

bool OutputLayer::UpdateOutput(mat input){
    
    mat pre_act = input * weight_;
    output_ = DComputeOutputFunc(pre_act, act_func_);
    return true;
}