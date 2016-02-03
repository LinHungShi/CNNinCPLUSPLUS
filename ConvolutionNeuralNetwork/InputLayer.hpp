//
//  InputLayer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//
#pragma once
#ifndef InputLayer_hpp
#define InputLayer_hpp
#include "InputLayer.hpp"
#include <stdio.h>

#endif /* InputLayer_hpp */

class InputLayer:public Layer{
public:
    InputLayer():Layer(0, "input"){}
    InputLayer(mat input): Layer(input, "input"){}
};
