//
//  FullLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/19/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "FullLayer.hpp"


void FullLayer::UpdateWeightGradient(mat input) {
  gradient_ = input.t() * delta_;
}

void FullLayer::UpdateWeight(double alpha) {
  weight_ = GradientDescent(alpha, weight_, gradient_);
}

bool FullLayer::UpdateOutput(mat input) {
  mat pre_act = input * weight_;
  
  output_ = act_func_->ComputeActFunc(pre_act);
  return true;
}

void FullLayer::DeleteWinitFunc() {
  w_init_func_ = nullptr;
  has_w_init_func_ = false;
}

void FullLayer::DeleteActFunc() {
  act_func_ = nullptr;
  has_act_func_ = false;
}

void FullLayer::InitWeight(int inp_dim) {
  weight_ = (*w_init_func_)(inp_dim, num_neuron_);
  is_w_init_ = true;
}

InitWeightFunction FullLayer::get_w_init_func() const {
  return  *w_init_func_;
}

void FullLayer::set_w_init_func( InitWeightFunction const&w_init_func) {
  w_init_func_ = new InitWeightFunction(w_init_func);
  
}

bool FullLayer::get_is_weight_init() const {
  return is_w_init_;
}

bool FullLayer::get_has_w_init_func() const {
  return has_w_init_func_;
}

bool FullLayer::get_has_act_func() const {
  return has_act_func_;
}

mat FullLayer::get_weight() const {
  return weight_;
}

mat FullLayer::get_gradient() const {
  return gradient_;
}

ActFunction FullLayer::get_act_func() const {
  return *act_func_;
}