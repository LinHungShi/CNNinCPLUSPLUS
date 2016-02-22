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
#include "Supplement.hpp"


class NeuronNet{

private:
    
    void FeedForward();
    void BackProp();
    vector<FullLayer*> layers_;
    mat input_, y_, output_;
    double alpha_;
    int epoch_;
    string init_method_, err_func_;
    
public:
    
    NeuronNet(mat input,
              mat y,
              vector<int> num_neuron,
              double alpha,
              string act_func,
              string out_func,
              string err_func,
              string init_method,
              int batch = 1,
              int epoch = 3);
    
    void AddLayer(FullLayer &layer);
    void InsertLayer(FullLayer &layer, int index);
    void DeleteLayer(int index);
    
    string CheckNNComplete();
    string CheckDimCompat();
    bool StopNN(string message);
    void TrainNN();
    mat get_output_(){return output_;}
    
    
};
#endif /* NeuralNetwork_hpp */