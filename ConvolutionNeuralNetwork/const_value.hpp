//
//  const_value.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 3/1/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef const_value_hpp
#define const_value_hpp
#include <stdio.h>
#include <iostream>
#define LONGLINE "-----------------------------------------"
inline void PrintError(double error) { std::cout << "Error: " << error << std::endl; }
template<class T>
inline void printValue(T t){ std::cout << t << std::endl; }
const int kLineWidth = 5;
const int kPrecision = 4;
const int kEpsilon = 0.01;
double g_alpha = 0.1;
int g_epoch = 10;
const std::string kUserDefinedMethod = "user_defined";
const std::string kHidLayer = "Hidlayer";
const std::string kOutputLayer = "OutputLayer";
const std::string kInputLayer = "InputLayer";
const std::string kNormLayer = "kNormLayer";

#endif /* const_value_hpp */
