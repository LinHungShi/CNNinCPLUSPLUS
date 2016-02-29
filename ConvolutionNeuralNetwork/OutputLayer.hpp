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
#include "err_function.hpp"
#include <iostream>
#include <iomanip>

class OutputLayer : public FullLayer{
  

private:
    
    bool UpdateDelta(mat y, ErrFunction &err_func);
    
public:

    OutputFunction output_func_;
    
    OutputLayer(int num_neuron,
                OutputFunction output_func):output_func_(output_func),
                                            FullLayer(
                                            num_neuron,
                                            "OutputLayer"){};
    bool UpdateParm(double alpha,
                   mat y,
                   mat input,
                   ErrFunction &err_func);
    
    bool UpdateOutput(mat input);
    
    void set_output_func_(OutputFunction &output_func){output_func_ = output_func;}
    
    friend ostream &operator<<(ostream &stream, OutputLayer &layer);
};
#endif /* OutputLayer_hpp */