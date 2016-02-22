//
//  OutputLayer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef OutputLayer_hpp
#define OutputLayer_hpp

#include <stdio.h>
#include "FullLayer.hpp"


class OutputLayer : public FullLayer{
  

private:
    
    bool UpdateDelta(mat y, string err_func);
    
public:

    OutputLayer(int num_neuron,
                int col,
                string init_method,
                string act_func,
                string name):FullLayer(num_neuron,
                                       col,
                                       init_method,
                                       act_func,
                                       name){}
    bool UpdateParm(double alpha,
                   mat y,
                   mat input,
                   string err_func);
    
    bool UpdateOutput(mat input);

};
#endif /* OutputLayer_hpp */