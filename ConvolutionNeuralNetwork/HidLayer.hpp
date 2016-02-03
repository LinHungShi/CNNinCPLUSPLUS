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
#endif /* HidLayer_hpp */

class HidLayer : public FullLayer{

protected:
    bool updateDelta(mat n_delta, mat n_weight);
    
public:
    HidLayer(int nn, mat w): FullLayer(nn, w){}
    HidLayer(int inp_dim, int nn, string init_method, string actfun, string name):FullLayer(inp_dim, nn, init_method, actfun, name){}
    bool updatePar(double alpha, mat y = randn<mat>(1,1), mat n_delta = randn<mat>(1,1), mat n_weight = randn<mat>(1,1));
};



