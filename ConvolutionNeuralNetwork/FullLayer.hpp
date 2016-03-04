//
//  FullLayer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/19/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef FullLayer_hpp
#define FullLayer_hpp

#include <stdio.h>
#include "base_layer.hpp"
#include "inl_nn_funcs.hpp"
#include "const_value.hpp"
#include "act_function.hpp"
#include "init_weight_function.hpp"


class FullLayer : public BaseLayer{
 protected:
  mat weight_, gradient_;
  bool is_w_init_;
  bool has_w_init_func_, has_act_func_;
  InitWeightFunction *w_init_func_ ;
  ActFunction *act_func_;
  void UpdateWeightGradient(mat input);
  
 public:
  // Constructor
  FullLayer(int num_neuron, string name):
    BaseLayer(num_neuron, name),
    is_w_init_(false),
    has_w_init_func_(false),
    has_act_func_(false),
    w_init_func_(nullptr),
    act_func_(nullptr) {};

  // Copy Constructor
  FullLayer(FullLayer const &copied): BaseLayer(copied) {
    weight_ = copied.get_weight();
    gradient_ = copied.get_gradient();
    has_w_init_func_ = copied.get_has_w_init_func();
    if(copied.get_has_w_init_func()) {
      InitWeightFunction w_init_func = copied.get_w_init_func();
      w_init_func_ = new InitWeightFunction(w_init_func);
    }
    
    if(copied.get_has_act_func()) {
      ActFunction act_func = copied.get_act_func();
      act_func_ = new ActFunction(act_func);
    }
  }
  // Destructor
  ~FullLayer() {
    cout << "call Full layer destructor ~" << endl;
    delete w_init_func_;
    delete act_func_;
  }
  
  // Manipulators
  void UpdateWeight(double alpha);
  virtual bool UpdateOutput(mat input);
  void DeleteWinitFunc();
  void DeleteActFunc();
  void InitWeight(int inp_dim);
  
  // Accessors and Setters
  InitWeightFunction get_w_init_func() const;
  // If user wants to delete weight initialization method, use DeleteWinitFunc() instead of its setter.
  void set_w_init_func(InitWeightFunction const &w_init_func);
  bool get_is_weight_init() const;
  bool get_has_w_init_func() const;
  bool get_has_act_func() const;
  mat get_weight() const;
  mat get_gradient() const;
  ActFunction get_act_func() const;
};



#endif /* FullLayer_hpp */


