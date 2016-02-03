//
//  HidLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/22/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "HidLayer.hpp"

bool HidLayer::updateDelta(mat n_delta, mat n_weight){

    mat der_output = diffAct(output, actfun);
    mat result = der_output % (n_delta * n_weight.t());
    delta = result;
    return true;
    
}

bool HidLayer::updatePar(double alpha, mat y, mat n_delta, mat n_weight){
    
    updateDelta(n_delta, n_weight);
    updateWGrad();
    updateW(alpha);
    return true;
}
