//
//  FullLayer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/19/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef FullLayer_hpp
#define FullLayer_hpp

#include <stdio.h>
#include "Layer.hpp"


class FullLayer : public Layer{
    
protected:
    
    int num_neuron;
    bool updateInput(mat &input);
    bool updateOutput();
    bool updateWGrad();

public:
    
    bool updateValue(mat input);
    
    FullLayer(int nn, mat weight):
                                num_neuron(nn),
                                Layer(weight, "weight"){}
    
    FullLayer(int inp_size, int nn, string init_method, string actfun, string name):
                                num_neuron(nn),
                                Layer(inp_size, nn,init_method, actfun, name){}
    
    virtual bool updatePar(double alpha,
                                mat y = randn<mat>(1,1),
                                mat n_delta = randn<mat>(1,1),
                                mat n_weight = randn<mat>(1,1)){return true;}
    
    
    
};

#endif /* FullLayer_hpp */


