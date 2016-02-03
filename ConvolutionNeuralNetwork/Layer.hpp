//
//  Layer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//
#pragma once
#ifndef Layer_hpp
#define Layer_hpp

#include <stdio.h>
#include <armadillo>
#include "Supplement.hpp"
#endif /* Layer_hpp */
WDQWDQWDDW
using namespace arma;
using namespace std;


class Layer{
    
protected:
    
    void updateW(double alpha){weight = weight - alpha * grad;}
    
    
public:
    
    string actfun, name;
    mat input, output, delta, grad, weight;
    Layer(mat value, string var_name);
    Layer(int inp_size, int nn, string init_method, string actfun, string name);
    
    virtual bool updateValue(mat input)=0;
    
};