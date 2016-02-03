
//
//  main.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include <iostream>
#include <armadillo>

#include "Layer.hpp"

#include "NeuralNetwork.hpp"
int main(int argc, const char * argv[]) {
    // insert code here...
    srand(int (time(0)));
    int num_obs = 1000;
    int inp_dim = 3;
    //int num_class = 2;
    double alpha = 0.00001;
    mat input(num_obs, inp_dim, fill::randn);
    mat y(num_obs,2, fill::zeros);
    y.submat(0, 0, 499, 0) = ones<colvec>(num_obs/2);
    y.submat(500, 1, 999, 1) = ones<colvec>(num_obs/2);
    //cout << "y" << y <<endl;
    vector<int> nn{50,30};
    string actfun = "tanh";
    string outfun = "softmax";
    string init_method = "randn";
    int epoch = 100;
    NeuronNet Nnet(input, y, nn, alpha, actfun, outfun, init_method, num_obs, epoch);
    Nnet.trainNN();
    /*HidLayer hid(inp_dim, nn, "randn", "sigmoid", "Hid");
    OutputLayer out(nn, num_class, "randn", "softmax", "Output");
    InputLayer inplayer(input);
    hid.updateValue(inplayer.input);
    out.updateValue(hid.output);
    double error = computeError(y, out.output);
    cout << "error:" << error << endl;
    for(int i = 0;i<=100;i++){
        out.updatePar(alpha, y);
        hid.updatePar(alpha, &out);
        hid.updateValue(inplayer.input);
        out.updateValue(hid.output);
       
        error = computeError(y, out.output);
        cout << "error:" << error << endl;
        //cout << out.output << endl;
        cout  << errorRate(y, out.output) << endl;
    }

    cout << "end" << endl;
    */

    
    
}
