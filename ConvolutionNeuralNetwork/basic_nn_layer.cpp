//
//  basic_nn_layer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/19/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "basic_nn_layer.hpp"


void BasicNNLayer::UpdateWeightGradient(mat input) {
  gradient_ = input.t() * delta_;
}

void BasicNNLayer::UpdateWeight(double alpha) {
  weight_ = GradientDescent(alpha, weight_, gradient_);
}

bool BasicNNLayer::UpdateOutput(mat input) {
  mat pre_act = input * weight_;
  
  output_ = act_func_->ComputeActFunc(pre_act);
  return true;
}

void BasicNNLayer::DeleteWinitFunc() {
  w_init_func_ = nullptr;
  has_w_init_func_ = false;
}

void BasicNNLayer::DeleteActFunc() {
  act_func_ = nullptr;
  has_act_func_ = false;
}

void BasicNNLayer::InitWeight(int inp_dim) {
  weight_ = (*w_init_func_)(inp_dim, num_neuron_);
  is_w_init_ = true;
}

InitWeightFunction BasicNNLayer::get_w_init_func() const {
  return  *w_init_func_;
}

void BasicNNLayer::set_w_init_func( InitWeightFunction const&w_init_func) {
  w_init_func_ = new InitWeightFunction(w_init_func);
  has_w_init_func_ = true;
  
}

bool BasicNNLayer::get_is_weight_init() const {
  return is_w_init_;
}

bool BasicNNLayer::get_has_w_init_func() const {
  return has_w_init_func_;
}

bool BasicNNLayer::get_has_act_func() const {
  return has_act_func_;
}

mat BasicNNLayer::get_weight() const {
  return weight_;
}

mat BasicNNLayer::get_gradient() const {
  return gradient_;
}

ActFunction BasicNNLayer::get_act_func() const {
  return *act_func_;
}