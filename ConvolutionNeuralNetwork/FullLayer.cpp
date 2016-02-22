//
//  FullLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/19/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "FullLayer.hpp"

bool FullLayer::UpdateWeightGradient(mat input)
{
    
    gradient_ = input.t() * delta_ ;
    return true;
    
}


