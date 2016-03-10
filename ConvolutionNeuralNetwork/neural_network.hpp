//
//  NeuralNetwork.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 1/29/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include "err_function.hpp"
#include "hidden_layer.hpp"
#include "output_layer.hpp"
#include "input_layer.hpp"
#include "init_weight_function.hpp"
class NeuronNet{
 private:
  vector<BaseLayer*> layers_;
  ErrFunction *err_func_;
  InitWeightFunction *w_init_func_;
  mat y_;
  bool has_w_init_func_;
  vector<HidLayer*> get_layers() const;
 
 public:
  // Constructors
  NeuronNet(mat const &input, mat const &y);
  
  ~NeuronNet() {
    delete err_func_;
    delete w_init_func_;
    for (BaseLayer* ptr : layers_) {
      delete ptr;
    }
  }
  
  // Manipulators
  
  // Insertion and Deletion
  
  // Insert hidden layer to layers_. If not specify the index, insert to the last
  void InsertLayer(BaseLayer &&layer);
  void InsertHidLayer(int index, HidLayer const&layer);
  
  // Delete hidden layer from layers_. If not specify the index, delete the last
  void DeleteHidLayer();
  void DeleteHidLayer(int index);
  
  // Initialize all layers' weights with w_init_func
  void InitAllLayerWeight(bool update_all);

  // Training functions
  void TrainNN();
  void FeedForward();
  void BackProp();
  
  // Micellaneous
  void ClearHidLayers();
  bool IsNNComplete() const;
  int GetLayersSize() const;
  bool HasWinitFunc() const;
  friend ostream &operator<<(ostream &, NeuronNet const&);
  // Since hidden layers are in vector, there is no accessor to use. Use these
  // two methods to get and set hidden layer
  HidLayer GetHidLayer(int index);
  void setHidLayer(HidLayer const&hid_layer, int index);
  
  // Accessors and Setters
  InitWeightFunction get_w_init_func() const;
  void set_w_init_func(InitWeightFunction const &w_init_func);
  ErrFunction get_err_func() const;
  void set_err_func(ErrFunction const &err_func);
  OutputLayer get_output_layer() const;
  void set_output_layer(OutputLayer const &output_layer);
  mat get_output() const;
  mat get_input() const;
  void set_input(mat const &input);
  mat get_y() const;
  void set_y(mat const &y);
  int get_epoch() const;
  void set_epoch(int epoch);
  double get_alpha() const;
  void set_alpha(double alpha);
};
#endif /* NeuralNetwork_hpp */