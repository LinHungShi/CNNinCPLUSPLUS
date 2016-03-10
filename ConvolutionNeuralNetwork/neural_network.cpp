//
//  NeuralNetwork.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 1/29/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "neural_network.hpp"

NeuronNet::NeuronNet(mat const &input, mat const &y) {
  InputLayer *ptr = new InputLayer(input);
  layers_.push_back(ptr);
  y_ = y;
}

void NeuronNet::InsertLayer(BaseLayer &&layer) {
  
  
  layers_.push_back(&layer);
    
}

void NeuronNet::InsertHidLayer(int index, HidLayer const&layer) {
  HidLayer  *ptr = new HidLayer(layer);
  layers_.insert(layers_.begin() + index, ptr);
}

void NeuronNet::DeleteHidLayer() {
  layers_.pop_back();
}

void NeuronNet::DeleteHidLayer(int index) {
  layers_.erase(layers_.begin() + index);
}

void NeuronNet::InitAllLayerWeight(bool update_all) {
  int inp_dim = (int)input_.n_cols;
  for (vector<HidLayer*>::iterator it=layers_.begin(); it!=layers_.end(); it++) {
    if(update_all && (**it).get_has_w_init_func()) {
      InitWeightFunction temp2 = (**it).get_w_init_func();
      (**it).set_w_init_func(*w_init_func_);
      (**it).InitWeight(inp_dim);
      (**it).set_w_init_func(temp2);
    } else if (!(**it).get_has_w_init_func()){
      (**it).set_w_init_func(*w_init_func_);
      (**it).InitWeight(inp_dim);
    }
    inp_dim = (**it).get_num_neuron();
  }
  if (update_all && output_layer_->get_has_w_init_func()) {
    InitWeightFunction temp = output_layer_->get_w_init_func();
    output_layer_->set_w_init_func(*w_init_func_);
    output_layer_->InitWeight(inp_dim);
    output_layer_->set_w_init_func(temp);
    
  } else if (!output_layer_->get_has_w_init_func()) {
    output_layer_->set_w_init_func(*w_init_func_);
    output_layer_->InitWeight(inp_dim);
  }
}

void NeuronNet::TrainNN() {
  double error;
  for (int i = 1; i <= epoch_; i++){
    FeedForward();
    BackProp();
    error = err_func_->ComputeErrFunc(output_, y_);
    PrintError(error);
    if(error < kEpsilon) {
      break;
    }
  }
  cout << output_;

}

void NeuronNet::FeedForward() {
  mat x = input_;
  for(vector<HidLayer*>::iterator it=layers_.begin(); it!=layers_.end(); ++it) {
    (*it)->UpdateOutput(x);
    x = (*it)->get_output();
  }
  output_layer_->UpdateOutput(x);
  this->output_ = output_layer_->get_output();
}

void NeuronNet::BackProp() {
  mat input = layers_[layers_.size()-1]->get_output();
  output_layer_->UpdateParm(alpha_, y_, input, *err_func_);
  int layer_size = GetLayersSize();
  mat next_layer_delta = output_layer_->get_delta();
  mat next_layer_weight = output_layer_->get_weight();
  
  for(int i=layer_size-1; i>0; i--) {
    (*layers_[i]).UpdateParm(alpha_, next_layer_delta,next_layer_weight, layers_[i-1]->get_output());
    next_layer_delta = layers_[i]->get_delta();
    next_layer_weight = layers_[i]->get_weight();
  }
  
  (*layers_[0]).UpdateParm(alpha_, next_layer_delta, next_layer_weight, input_);
}

void NeuronNet::ClearHidLayers() {
  layers_.clear();
}

// Wait for modification
bool NeuronNet::IsNNComplete() const {
    return layers_.empty();
}

int NeuronNet::GetLayersSize() const {
    return (int)layers_.size();
}

bool NeuronNet::HasWinitFunc() const {
  return has_w_init_func_;
}

HidLayer NeuronNet::GetHidLayer(int index) {
  return *layers_[index];
}

void NeuronNet::setHidLayer(const HidLayer &hid_layer, int index){
  HidLayer *ptr = new HidLayer(hid_layer);
  layers_[index] = ptr;
}

InitWeightFunction NeuronNet::get_w_init_func() const {
  return *w_init_func_;
}

void NeuronNet::set_w_init_func(InitWeightFunction const &w_init_func) {
  w_init_func_ = new InitWeightFunction(w_init_func);
}

ErrFunction NeuronNet::get_err_func() const {
  return *err_func_;
}

void NeuronNet::set_err_func(const ErrFunction &err_func) {
  err_func_ = new ErrFunction(err_func);
}
OutputLayer NeuronNet::get_output_layer() const {
  OutputLayer out = *output_layer_;
  return out;
}

void NeuronNet::set_output_layer(OutputLayer const &output_layer) {
  output_layer_ = new OutputLayer(output_layer);
}

mat NeuronNet::get_output() const {
  return output_;
}

mat NeuronNet::get_input() const{
  return input_;
}

void NeuronNet::set_input(mat const &input) {
  input_ = input;
}

mat NeuronNet::get_y() const {
  return y_;
}

void NeuronNet::set_y(mat const &y){
  y_ = y;
}

int NeuronNet::get_epoch() const {
  return epoch_;
}

void NeuronNet::set_epoch(int epoch) {
  epoch_ = epoch;
}

double NeuronNet::get_alpha() const {
  return alpha_;
}

void NeuronNet::set_alpha(double alpha) {
  alpha_ = alpha;
}

vector<HidLayer*> NeuronNet::get_layers() const {
  return layers_;
}

ostream &operator<<(ostream &stream, NeuronNet const &nnet) {
  int width = kLineWidth;
  stream << "Neural Net..." << endl;
  stream << "Epoch: " << setw(width) << nnet.epoch_ << endl;
  stream << "Learning Rate: " << nnet.alpha_ << endl;
  stream << "Number of Layers: " << nnet.layers_.size() << endl;
  if(nnet.HasWinitFunc()) {
    stream << "Universal Layer Weight Initialization Method: " << setw(width) << nnet.get_w_init_func().get_method_name() << endl;
  } else {
      stream << "All Layer Weight Initialization Method: " << setw(width) << "None" << endl;
  }
  
  stream << "All layers' information are printed..." << endl;
  stream << LONGLINE << endl;
  //vector<HidLayer*>::iterator it = nnet.get_layers().begin();
  //int num = 0;
  stream << "Output Layer: " << setw(width) << endl;

  stream << *nnet.output_layer_ << endl;
  
  /*for(;it!=nnet.layers_.end();++it) {
    stream << " Hidden Layer: " << setw(width) << num << endl;
    stream << **it << endl;
    ++num;
  }*/
  int size = nnet.GetLayersSize();
  for (int i =0; i < size; i++) {
    stream << "Hidden Layer: " << setw(width) << i << endl;
    stream << *(nnet.layers_[i]) << endl;
  }
  return stream;
}

