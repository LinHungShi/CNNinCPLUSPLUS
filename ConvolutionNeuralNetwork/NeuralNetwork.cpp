//
//  NeuralNetwork.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 1/29/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "NeuralNetwork.hpp"




/*NeuronNet::NeuronNet(mat input,
                     mat y,
                     vector<int> num_neuron,
                     double alpha,
                     string act_func,
                     string out_func,
                     string err_func,
                     string init_method,
                     int batch,
                     int epoch)
{
    if(act_func == "softmax")
        err_func_ = "crossentropy";
    err_func_ = err_func;
    input_ = input;
    init_method_ = init_method;
    epoch_ = epoch;
    y_ = y;
    alpha_ = alpha;
    int length = (int)num_neuron.size();
    int input_dim = (int)input.n_cols;

    for (int i = 0; i < length; i++) {
        
        HidLayer* hidlayer = new HidLayer(input_dim, num_neuron[i], init_method_, act_func, "Hid");
        input_dim = num_neuron[i];
        
        layers_.push_back(hidlayer);
    }
   
    int num_class = (int) y.n_cols;
    OutputLayer* outlayer = new OutputLayer(num_neuron[length - 1], num_class, init_method, out_func, "Output");
    layers_.push_back(outlayer);
    
}
*/


void NeuronNet::InsertLayer(FullLayer &layer)
{

    layers_.push_back(&layer);
    
}

void NeuronNet::InsertLayer(FullLayer &layer, int index)
{
    
    layers_.insert(layers_.begin() + index, &layer);

}

void NeuronNet::DeleteFirstLayer()
{
    
    layers_.erase(layers_.begin());
    
}

void NeuronNet::DeleteLastLayer()
{
    layers_.erase(layers_.end());
    
}

void NeuronNet::DeleteLayer(int index)
{
    
    layers_.erase(layers_.begin() + index);
    
}

void NeuronNet::ClearLayers()
{
    
    layers_.clear();
    
}

bool NeuronNet::IsLayersEmpty()
{

    return layers_.empty();

}

int NeuronNet::GetLayersSize()
{
    
    return (int)layers_.size();


}

bool NeuronNet::CheckNNComplete(){
    
    if( (dynamic_cast<HidLayer*>(layers_[0])) && (dynamic_cast<OutputLayer*>(layers_.back())))
       {
           
           return true;
    
       }
       
       return false;
}




void NeuronNet::FeedForward()
{
    
    mat x = input_;
    for(vector<FullLayer*>::iterator it = layers_.begin();it != layers_.end();++it)
    {
        
        
        (*it)->UpdateOutput(x);
        x = (*it)->output_;
        
    }
    
    int output_index = (int)layers_.size()-1;
    this->output_ = layers_[output_index]->output_;
}

void NeuronNet::BackProp()
{

    int output_index = (int)(layers_.size()-1);
    mat input = layers_[output_index - 1]->output_;
    UpdateOutputLayerParm(*layers_[output_index], y_, input, alpha_, err_func_);
    
    
    for(int i = output_index - 1; i > 0; i--)
    {

        mat next_layer_delta = layers_[i+1]->delta_;
        mat next_layer_weight = layers_[i+1]->weight_;
        input = layers_[i-1]->output_;
        UpdateHidLayerParm(*layers_[i], next_layer_delta,next_layer_weight, alpha_, input);
        
    }
    
    mat next_layer_delta = layers_[1]->delta_;
    mat next_layer_weight = layers_[1]->weight_;
    input = this->input_;
    UpdateHidLayerParm(*layers_[0], next_layer_delta, next_layer_weight, alpha_, input);
    
}

void NeuronNet::InitAllLayerWeight(bool update_all)
{
 
    int inp_dim = (int)input_.n_cols;
    
    for (vector<FullLayer*>::iterator it=layers_.begin(); it!=layers_.end(); it++)
    {
        if((**it).has_w_init_func_ == true)
        {
        
            InitWeightFunction temp = *(**it).w_init_func_;
            (*it)->w_init_func_ = w_init_func_;
            (*it)->InitWeight(inp_dim);
            (**it).w_init_func_ = &temp;
            inp_dim = (*it)->num_neuron_;
        
        }
        
        (*it)->w_init_func_ = w_init_func_;
        (*it)->InitWeight(inp_dim);
        inp_dim = (*it)->num_neuron_;
        
    }
    
}

void NeuronNet::TrainNN()
{

    double error;
    for (int i = 1; i <= epoch_; i++)
    {
        
        FeedForward();
        BackProp();
        error = err_func_.ComputeErrFunc(output_, y_);
        cout << "error is:" << error << endl;
        if(error < 0.01)
            break;
    }
    cout << output_;
}

ostream &operator<<(ostream &stream, NeuronNet &nnet)
{

    int width = 5;
    stream << "Neural Net..." << endl;
    stream << "Epoch: " << setw(width) << nnet.epoch_ << endl;
    stream << "Learning Rate: " << nnet.alpha_ << endl;
    stream << "Number of Layers: " << nnet.layers_.size() << endl;
    if(nnet.has_w_init_func_)
    {
    
        stream << "All Layer Weight Initialization Method: " << setw(width) << nnet.w_init_func_->init_method_name_ << endl;
    
    }
    else
    {
        
        stream << "All Layer Weight Initialization Method: " << setw(width) << "None" << endl;
        
    }
    
    stream << "All layers' information are printed..." << endl;
    stream << PrintSepLine() << endl;
    vector<FullLayer*>::iterator it=nnet.layers_.begin();
    int num = 0;
    
    for(;it!=nnet.layers_.end()-1;++it)
    {
        
        stream << "Layer: " << setw(width) << num << endl;
        HidLayer* hid_layer = dynamic_cast<HidLayer*>(*it);
        stream << *hid_layer << endl;
        ++num;
        
    }
    stream << "Output Layer: " << endl;
    OutputLayer *output_layer = dynamic_cast<OutputLayer*>(nnet.layers_[num]);
    stream << *output_layer << endl;
    return stream;

}

