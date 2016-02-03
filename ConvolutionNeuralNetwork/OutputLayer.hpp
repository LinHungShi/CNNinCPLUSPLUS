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
#endif /* OutputLayer_hpp */

class OutputLayer : public FullLayer{
  

private:
    bool updateDelta(mat y);
    
protected:
    
public:
    OutputLayer(int nn, mat w): FullLayer(nn, w){}
    OutputLayer(int nn, int col, string init_method, string actfun, string name):FullLayer(nn, col, init_method, actfun, name){}
    bool updatePar(double alpha, mat y = randn<mat>(1,1), mat n_delta = randn<mat>(1,1), mat n_weight = randn<mat>(1,1));

};