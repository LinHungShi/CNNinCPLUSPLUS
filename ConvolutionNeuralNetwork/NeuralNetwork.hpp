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
#include "OutputLayer.hpp"
#include "HidLayer.hpp"
#endif /* NeuralNetwork_hpp */

class NeuronNet{

private:
    void feedForward();
    void backProp();
    vector<FullLayer*> layers;
    mat input;
    mat y;
    mat output;
    double alpha;
    int epoch;
    string init_method;
public:
    NeuronNet(mat input, mat y, vector<int> nn, double alpha,  string actfun, string outfun, string init_method, int batch = 1, int epoch = 3);
    void trainNN();
    mat getOutput(){return this->output;}
    
};
