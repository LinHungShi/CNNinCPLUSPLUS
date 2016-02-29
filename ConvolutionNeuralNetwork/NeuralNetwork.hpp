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
#include "err_function.hpp"
#include "init_weight_function.hpp"
class NeuronNet{

public:
    
    vector<FullLayer*> layers_;
    
    ErrFunction err_func_;
    
    mat input_,
        y_,
        output_;
    
    double alpha_;
    int epoch_;

    
    void FeedForward();
    void BackProp();
    InitWeightFunction *w_init_func_;
    bool has_w_init_func_;
    
    

    
    /*NeuronNet(mat input,
              mat y,
              vector<int> num_neuron,
              double alpha,
              string act_func,
              string out_func,
              string err_func,
              string init_method,
              int batch = 1,
              int epoch = 3);
     */
    NeuronNet(mat input,
              mat y,
              ErrFunction err_func,
              double alpha,
              int epoch,
              string init_method):input_(input),
                                  y_(y),
                                  err_func_(err_func),
                                  epoch_(epoch),
                                  alpha_(alpha),
                                  has_w_init_func_(false){};
    ~NeuronNet()
    {
        
        delete w_init_func_;
    
    }
    
    void InsertLayer(FullLayer &layer);
    void InsertLayer(FullLayer &layer, int index);
    void DeleteLastLayer();
    void DeleteFirstLayer();
    void DeleteLayer(int index);
    void ClearLayers();
    bool IsLayersEmpty();
    int GetLayersSize();

    void set_w_init_func_(InitWeightFunction *w_init_func)
    {
        
        w_init_func_ = w_init_func;
        has_w_init_func_ = true;
        
    }
    
    void InitAllLayerWeight(bool update_all);
    
    void TrainNN();

    bool CheckNNComplete();
    bool StopNN(string message);

    FullLayer get_layers_(int index){return *(layers_[index]);};
    mat get_output_(){return output_;}
    
    friend ostream &operator<<(ostream &, NeuronNet &);
};
#endif /* NeuralNetwork_hpp */