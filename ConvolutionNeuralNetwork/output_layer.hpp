//
//  OutputLayer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef output_layer_hpp
#define output_layer_hpp

#include <stdio.h>
#include "basic_nn_layer.hpp"
#include "err_function.hpp"
#include <iostream>
#include <iomanip>

class OutputLayer : public BasicNNLayer{
 private:
  bool UpdateDelta(mat const &y, ErrFunction const &err_func);
    
 public:
  // Constructors
  OutputLayer(int num_neuron): BasicNNLayer(num_neuron, kOutputLayer) {};
  
  ~OutputLayer(){
    cout << "call output layer destructor" << endl;
    //delete act_func_;
  }
  
  // Maniulators
  bool UpdateParm(double alpha, mat const &y, mat const &input, ErrFunction const &err_func);
  friend ostream &operator<<(ostream &stream, OutputLayer const &layer);
  
  //Accessors and Setters
  // act_func's Getter is defined in FullLayer
  // User must set is_hid in act_func as false, otherwise refuse to set activation
  // fucntion in output layer
  void set_act_func(ActFunction &act_func);
};
#endif /* OutputLayer_hpp */