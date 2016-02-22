//
//  ActFuncOp.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/16/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef ActFuncOp_hpp
#define ActFuncOp_hpp
#include <armadillo>
#include <stdio.h>
#include <iostream>
#include <map>
using namespace arma;
using namespace std;
mat DComputeActFunc(mat, string);
mat DComputeOutputFunc(mat, string);
mat DDiffActFunc(mat, string);
mat DDiffOutputFunc(mat, string);
double DComputeErrFunc(mat y, mat pred, string errfunc);
mat DDiffErrFunc(mat pred, mat y, string errfunc);

#endif /* ActFuncOp_hpp */
