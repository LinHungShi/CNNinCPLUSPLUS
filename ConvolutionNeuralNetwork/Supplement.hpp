//
//  Supplement.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#ifndef Supplement_hpp
#define Supplement_hpp

#include <stdio.h>
#include <armadillo>
using namespace arma;
using namespace std;
#endif /* Supplement_hpp */

mat actNeuron(mat pre_act, string actfun, string l_name);
mat diffAct(mat pre_act, string actfun);
mat initWeight(int row, int col, string init_method);
double computeError(mat y, mat predict);
double errorRate(mat y, mat predict);