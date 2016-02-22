//
//  Layer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/18/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <stdio.h>
#include <armadillo>

using namespace arma;
using namespace std;


class Layer{
    
protected:
    
    void UpdateWeight(double alpha)
    {
        
        weight_ = weight_ - alpha * gradient_;
    
    }
   
public:
    
    string act_func_, name_, num_neuron_;
    mat output_, delta_, gradient_, weight_;

    Layer(int input_size,
          int num_neuron,
          string init_method,
          string act_func,
          string name);
    
    void InitWeight(int row,
                    int col,
                    string init_method);
 
};

#endif /* Layer_hpp */