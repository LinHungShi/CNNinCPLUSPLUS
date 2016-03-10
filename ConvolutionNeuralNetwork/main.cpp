
//
//  main.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include <iostream>
#include <armadillo>
#include "neural_network.hpp"

int main(int argc, const char * argv[]) {
  
    srand(int (time(0)));
    int num_obs = 1000;
    int inp_dim = 500;
    //int num_class = 2;
    double alpha = 0.0005;
    mat input(num_obs, inp_dim, fill::randn);
    mat y(num_obs,2, fill::zeros);
    y.submat(0, 0, 499, 0) = ones<colvec>(num_obs/2);
    y.submat(500, 1, 999, 1) = ones<colvec>(num_obs/2);
    //cout << "y" << y <<endl;
    vector<int> nn{150,100};
    int classes = 2;
    string actfun = "tanh";
    string outfun = "softmax";
    string init_method = "randn";
    //string errfunc = "crossentropy";
    int epoch = 1000;
    bool is_hid = true;
    bool not_hid = false;
  
    ActFunction hidact(actfun, is_hid);
    auto tanh = [](mat pre_act)->mat{return 1/(1 + exp(-pre_act));};
    auto diff_tanh = [](mat act)->mat{return 1 - pow(act, 2);};
    ActFunction hid_act_tanh(tanh, diff_tanh, is_hid);
    HidLayer hidlayer(nn[0]);
    HidLayer hidlayer2(nn[1]);
    hidlayer.set_act_func(hidact);
    hidlayer2.set_act_func(hid_act_tanh);
  
    ActFunction outact(outfun, not_hid);
    OutputLayer outlayer(classes);
    outlayer.set_act_func(outact);
  
    ErrFunction errfunc("crossentropy");
    NeuronNet NN(alpha, epoch, input, y);
    InitWeightFunction w_init_func(init_method);
    NN.set_w_init_func(w_init_func);
  NN.InsertHidLayer(std::move(hidlayer));
  NN.InsertHidLayer(std::move(hidlayer2));
  NN.set_output_layer(std::move(outlayer));
  NN.set_err_func(errfunc);
    NN.InitAllLayerWeight(true);
    cout << NN << endl;
    NN.TrainNN();
}
