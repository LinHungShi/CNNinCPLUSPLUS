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
#include "hidden_act_function.hpp"
#include "ActFuncOp.hpp"
#include <iomanip>


class HidLayer : public FullLayer{

protected:
    
    bool UpdateDelta(mat next_layer_delta, mat next_layer_weight);
    
public:
    
    HidActFunction act_func_;
    
    HidLayer(int num_neuron,
             HidActFunction &act_func):act_func_(act_func),
                                       FullLayer(
                                       num_neuron,
                                       "HidLayer"){};

    bool UpdateParm(double alpha,
                    mat next_layer_delta,
                    mat next_layer_weight,
                    mat input);
    
    bool UpdateOutput(mat input);
    void set_act_func_(HidActFunction &act_func){act_func_ = act_func;}
    
    friend ostream &operator<<(ostream &stream, HidLayer &layer);
};

#endif /* HidLayer_hpp */

