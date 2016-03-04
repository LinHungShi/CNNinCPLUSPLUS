//
//  HidLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/22/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "HidLayer.hpp"

bool HidLayer::UpdateDelta(mat const &next_layer_delta, mat const &next_layer_weight) {
  
  mat der_output = act_func_->DiffActFunc(output_);
  mat result = der_output % (next_layer_delta * trans(next_layer_weight));
  delta_ = result;
  return true;
}

bool HidLayer::UpdateParm(double alpha,mat next_layer_delta,
                          mat next_layer_weight, mat input) {

  UpdateDelta(next_layer_delta, next_layer_weight);
  UpdateWeightGradient(input);
  UpdateWeight(alpha);
  return true;
}

void HidLayer::set_act_func(const ActFunction &act_func) {
  if(act_func.get_is_hid()) {
    act_func_ = new ActFunction(act_func);
    has_act_func_ = true;
  } else {
    cout << "is_hid must be true, insertion fails" << endl;
  }
}

ostream &operator<<(ostream &stream, HidLayer const &layer) {
  int width = kLineWidth;
  int prec = kPrecision;
  stream << "Layer name: " << setw(width) << layer.name_ << endl;
  stream << "Number of Neuron:" << setw(width) << layer.num_neuron_ << endl;
  stream << "Activation Function: " << setw(width) << layer.act_func_->get_func_name() << endl;
    
  if(layer.has_w_init_func_) {
    stream << "Weight Initialization: " << setw(width) << layer.w_init_func_->get_method_name() <<endl;
  } else {
      stream << "Weight Initialization: " << setw(width) << "None"<< endl;
  }

  stream << setprecision(prec) << "Weight Dimension: " << setw(width) << "(" << layer.weight_.n_rows << "," << layer.weight_.n_cols << ")" << endl;
    
  stream << LONGLINE << endl;
  return stream;
    
}