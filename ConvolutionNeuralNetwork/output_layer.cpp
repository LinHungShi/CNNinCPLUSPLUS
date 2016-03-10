//
//  OutputLayer.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "output_layer.hpp"


//bool OutputLayer::updatePar(double alpha, Layer const *n_layer, Layer const *p_layer){
    
    
//    cout << "wrong function" << endl;
//    return true;
//}

bool OutputLayer::UpdateDelta(mat const &y, ErrFunction const &err_func) {

  delta_ = err_func.DiffErrFunc(output_, y, *act_func_);
  return true;
}

bool OutputLayer::UpdateParm(double alpha,
                             mat const &y,
                             mat const &input,
                             ErrFunction const &err_func) {

  UpdateDelta(y, err_func);
  UpdateWeightGradient(input);
  UpdateWeight(alpha);
  return true;
}

void OutputLayer::set_act_func(ActFunction &act_func) {
  if(!act_func.get_is_hid()) {
    act_func_ = new ActFunction(act_func);
    has_act_func_ = true;
  } else {
    cout << "is_hid must be false, insertion fails" << endl;
  }
}

ostream &operator<<(ostream &stream, OutputLayer const &layer)
{
  int width = kLineWidth;
  int prec = kPrecision;
  cout << layer.name_;
  stream << "Layer name: " << setw(width) << layer.name_ << endl;
  stream << "Number of Neuron:" << setw(width) << layer.num_neuron_ << endl;
  stream << "Output Function: " << setw(width) << layer.act_func_->get_func_name() << endl;
    
  if(layer.has_w_init_func_) {
        stream << "Weight Initialization: " << setw(width) << layer.w_init_func_->get_method_name();
  } else {
      stream << "Weight Initialization: " << setw(width) << "None" << endl;;
    }
    
  stream << setprecision(prec) << "Weight Dimension: " << setw(width) << "(" << layer.weight_.n_rows << "," << layer.weight_.n_cols << ")" << endl;
    
  stream << LONGLINE << endl;
  return stream;
    
}