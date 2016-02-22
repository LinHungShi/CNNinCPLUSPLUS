
//
//  main.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include <iostream>
#include <armadillo>
#include "NeuralNetwork.hpp"

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
    string actfun = "tanh";
    string outfun = "softmax";
    string init_method = "randn";
    string errfunc = "crossentropy";
    int epoch = 1000;
    NeuronNet Nnet(input, y, nn, alpha, actfun, outfun, errfunc, init_method, num_obs, epoch);
    Nnet.TrainNN();
    }
