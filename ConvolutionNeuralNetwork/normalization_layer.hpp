//
//  normalization_layer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 3/9/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef normalization_layer_hpp
#define normalization_layer_hpp

#include <stdio.h>
#include <armadillo>
#include "base_layer.hpp"
#include "act_function.hpp"

using namespace std;
using namespace arma;

class NormLayer : public BaseLayer {
 private:
  ActFunction *act_func_;
};
#endif /* normalization_layer_hpp */
