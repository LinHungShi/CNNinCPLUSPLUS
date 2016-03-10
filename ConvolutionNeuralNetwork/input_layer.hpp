//
//  InputLayer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 3/10/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef input_layer_hpp
#define input_layer_hpp

#include <stdio.h>
#include <armadillo>
#include "base_layer.hpp"
#include "const_value.hpp"
class InputLayer : public BaseLayer {
 private:
  mat input_;
  
 public:
  InputLayer(mat input):BaseLayer(0, kInputLayer), input_(input){};
  void updateParam() override{};
};

#endif /* InputLayer_hpp */
