//
//  Layer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//


#include "Layer.hpp"


Layer::Layer(int inp_size, int nn, string init_method, string actfun, string var_name){
    this->weight = initWeight(inp_size, nn, init_method);
    this->actfun = actfun;
    this->name = var_name;
}
Layer::Layer(mat value, string var_name){
    if(var_name =="weight")
        weight = value;
    else if(var_name == "input"){
        this -> input = value;
    }
}