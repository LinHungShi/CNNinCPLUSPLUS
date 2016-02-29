//
//  FullLayer.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/19/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef FullLayer_hpp
#define FullLayer_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "ActFuncOp.hpp"
#include "init_weight_function.hpp"
#include "line_print_func.hpp"

class FullLayer : public Layer{
    
protected:
    
    bool UpdateWeightGradient(mat input);

public:
    
    mat weight_, gradient_;
    bool is_weight_init_;
    InitWeightFunction *w_init_func_;
    bool has_w_init_func_;
    
    virtual bool UpdateOutput(mat input)=0;
    
    FullLayer(int num_neuron,
              string name):Layer(num_neuron, name), is_weight_init_(false), has_w_init_func_(false){};
    
    
    
    void InitWeight(int inp_dim);
    
    void set_w_init_func_(InitWeightFunction w_init_func)
    {
        
        w_init_func_ = &w_init_func;
        has_w_init_func_ = true;
    
    }
    
    void UpdateWeight(double alpha)
    {
        
        weight_ = weight_ - alpha * gradient_;
        
    }
};



#endif /* FullLayer_hpp */


