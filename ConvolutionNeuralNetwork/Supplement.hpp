//
//  Supplement.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef Supplement_hpp
#define Supplement_hpp

#include <stdio.h>
#include <armadillo>
#include "HidLayer.hpp"
#include "OutputLayer.hpp"
using namespace arma;
using namespace std;

void UpdateHidLayerParm(FullLayer &layer,
                      mat next_layer_delta,
                      mat n_layer_weight,
                      double alpha,
                      mat input);

void UpdateOutputLayerParm(FullLayer &layer,
                           mat y,
                           mat input,
                           double alpha,
                           ErrFunction &err_func);
#endif /* Supplement_hpp */