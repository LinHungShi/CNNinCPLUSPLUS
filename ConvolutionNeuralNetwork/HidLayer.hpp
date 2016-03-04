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
#include "inl_nn_funcs.hpp"
#include "const_value.hpp"
#include <iomanip>

class HidLayer : public FullLayer{
 private:
  bool UpdateDelta(mat const &next_layer_delta, mat const &next_layer_weight);
    
public:
  // Constructor
  HidLayer(int num_neuron):FullLayer(num_neuron, kHidLayer){};
  
  ~HidLayer(){
    cout << "call hidden layer destructor" << endl;
    //delete act_func_;
  }
  // Manipulators
  bool UpdateParm(double alpha,
                  mat next_layer_delta,
                  mat next_layer_weight,
                  mat input);
  
  //HidLayer Printing Method
  friend ostream &operator<<(ostream &stream, HidLayer const &layer);
  
  //Accessors and Setters
  // act_func's Getter is defined in FullLayer
  // User must set is_hid in act_func as true, otherwise refuse to set activation
  // fucntion in hidden layer
  void set_act_func(ActFunction const &act_func);
};

#endif /* HidLayer_hpp */

