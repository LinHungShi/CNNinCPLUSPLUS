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
#include "ActFuncOp.hpp"

class FullLayer : public Layer{
    
protected:
    
    int num_neuron_;
    bool UpdateWeightGradient(mat input);

public:
    
    virtual bool UpdateOutput(mat input)=0;
    
    FullLayer(int input_size,
              int num_neuron,
              string init_method,
              string act_func,
              string name):Layer(input_size,
                                 num_neuron,
                                 init_method,
                                 act_func,
                                 name){}
    
};

#endif /* FullLayer_hpp */


