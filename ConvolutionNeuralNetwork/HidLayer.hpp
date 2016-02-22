//
//  HidLayer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/22/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef HidLayer_hpp
#define HidLayer_hpp

#include <stdio.h>
#include "FullLayer.hpp"


class HidLayer : public FullLayer{

protected:
    
    bool UpdateDelta(mat next_layer_delta, mat next_layer_weight);
public:
    
    HidLayer(int input_dim,
             int num_neuron,
             string init_method,
             string act_func,
             string name):FullLayer(input_dim,
                                    num_neuron,
                                    init_method,
                                    act_func,
                                    name){}

    bool UpdateParm(double alpha,
                    mat next_layer_delta,
                    mat next_layer_weight,
                    mat input);
    
    bool UpdateOutput(mat input);
};

#endif /* HidLayer_hpp */

