//
//  default_weight_init_funcs.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/23/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "default_weight_init_funcs.hpp"


typedef mat(*WeightInit)(int row, int col);

mat Randn(int, int);
mat Radnu(int, int);
mat Ones(int, int);
map<string, WeightInit> CreateWInitLookUpTable();
static map<string, WeightInit> weight_init_func_look_up_table = CreateWInitLookUpTable();



mat Randn(int row, int col)
{
    
    return randn<mat>(row, col);

}

mat Randu(int row, int col)
{
    
    return randu<mat>(row,col);
    
}

mat Ones(int row, int col)
{
    
    return ones(row, col);
    
}

map<string, WeightInit> CreateWInitLookUpTable()
{
    
    map<string, WeightInit> look_up_table;
    look_up_table["randn"] = Randn;
    look_up_table["randu"] = Randu;
    look_up_table["ones"] = Ones;
    return look_up_table;
    
}

mat InitWeight(int row, int col, string init_method_name)
{
 
    return weight_init_func_look_up_table[init_method_name](row, col);
    
}
