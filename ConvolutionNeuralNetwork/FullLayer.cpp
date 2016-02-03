//
//  FullLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/19/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "FullLayer.hpp"


bool FullLayer::updateValue(mat input){
    
    updateInput(input);
    updateOutput();
    return true;
}

bool FullLayer::updateInput(mat &input){
    
 
    if((this->input.is_empty() == 0) && accu(this->input - input) == 0)
        return false;
    this->input = input;
    return true;
    
}

bool FullLayer::updateOutput(){
    
    mat pre_act = input * weight;
    
    mat act = actNeuron(pre_act, actfun, name);
    
    output = act;
    
    return true;
}

bool FullLayer::updateWGrad(){
    
    grad = weight;
    for(int i = 0;i<grad.n_rows;i++){
        grad.row(i) = input.col(i).t() * delta;
    }
    
    return true;
}