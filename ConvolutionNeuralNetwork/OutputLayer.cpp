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

bool OutputLayer::updatePar(double alpha, mat y, mat n_delta, mat n_weight){

    updateDelta(y);
    updateWGrad();
    //cout << weight << endl;
    updateW(alpha);
    //cout << weight << endl;
    return true;
}

bool OutputLayer::updateDelta(mat y){

    if(actfun == "softmax"){
        delta = output - y;
    }
    return true;
}