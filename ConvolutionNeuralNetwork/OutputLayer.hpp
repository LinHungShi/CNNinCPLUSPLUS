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
  bool UpdateDelta(mat const &y, ErrFunction const &err_func);
    
 public:
  // Constructors
  OutputLayer(int num_neuron): FullLayer(num_neuron, kOutputLayer) {};
  
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