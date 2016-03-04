//
//  default_weight_init_funcs.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/23/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "init_weight_function.hpp"

map<string, WeightInit> CreateWInitLookUpTable();

static map<string, WeightInit> weight_init_func_look_up_table = CreateWInitLookUpTable();

map<string, WeightInit> CreateWInitLookUpTable() {
  map<string, WeightInit> look_up_table;
  look_up_table["randn"] = Randn;
  look_up_table["randu"] = Randu;
  look_up_table["ones"] = Ones;
  return look_up_table;
}

mat InitWeight(int row, int col, string init_method_name) {
  return weight_init_func_look_up_table[init_method_name](row, col);
}