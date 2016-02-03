//
//  NeuralNetwork.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 1/29/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "NeuralNetwork.hpp"
#include <vector>
NeuronNet::NeuronNet(mat input, mat y, vector<int> nn, double alpha, string actfun, string outfun, string init_method, int batch, int epoch){
    this->input = input;
    this->init_method = init_method;
    this->epoch = epoch;
    this->y = y;
    this->alpha = alpha;
    int length = (int)nn.size();
    int inp_col = (int)input.n_cols;
    for (int i = 0; i < length; i++) {
        
        HidLayer* hidlayer = new HidLayer(inp_col, nn[i], init_method, actfun, "Hid");
        inp_col = nn[i];
        //cout << "weight:(" << hidlayer->weight.n_rows << "," << hidlayer->weight.n_cols << endl;
        layers.push_back(hidlayer);
    }
   
    int num_class = (int) y.n_cols;
    OutputLayer* outlayer = new OutputLayer(nn[length - 1], num_class, init_method, outfun, "Output");
    layers.push_back(outlayer);
    
}


void NeuronNet::feedForward(){
    mat x = input;
    
    for(vector<FullLayer*>::iterator it = layers.begin();it != layers.end();++it){
        (*it)->updateValue(x);
        x = (*it)->output;
        //cout << (*it)->weight(1,1) << endl;
    }
    
    int out_index = (int)layers.size()-1;
    this->output = layers[out_index]->output;
    
}

void NeuronNet::backProp(){
    
    //output layer updates parameters
    int out_index = (int)(layers.size()-1);
    
    layers[out_index]->updatePar(this->alpha, y);
    
    mat dummy;
    mat nl_delta = layers[out_index]->delta;
    mat nl_weight = layers[out_index]->weight;
    //hidden layers update parameters
    vector<FullLayer*>::reverse_iterator it = layers.rbegin();
    //cout << "--------------backprop-------------------" << endl;
    for (advance(it, 1);it != layers.rend(); ++it) {
        //cout << "number of neurons:" << (*it)->weight.n_cols << endl;
        (*it)->updatePar(alpha, dummy, nl_delta, nl_weight);
        nl_delta = (*it)->delta;
        nl_weight = (*it)->weight;
     //   cout << (*it)->weight(1,1) << endl;
    }
    //cout << "-------------------end----------------------" << endl;
}

void NeuronNet::trainNN(){
    double error;
    feedForward();
    //cout << "input:" << input << endl;
    error = computeError(y, this->output);
    for (int i = 1; i <= epoch; i++) {
        backProp();
        feedForward();
        //cout << "output:" << output<< endl;
        //cout << "y:" << y << endl;
        error = computeError(y, this->output);
        cout << "error is:" << error << endl;
    }
}