//
//  ActFuncOp.hpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/16/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#ifndef inl_nn_funcs_hpp
#define inl_nn_funcs_hpp

#include <armadillo>
#include <stdio.h>
#include <iostream>
#include <map>

class ActFunction;

using namespace arma;
using namespace std;

typedef mat (*FnPtr)(mat const &input);
typedef double (*ErrFuncPtr)(mat const &pred, mat const &y);
typedef mat (*DiffErrFuncPtr)(mat const&, mat const&, ActFunction const&);

static const map<string, FnPtr> CreateActFuncLookUpTable();
static const map<string, FnPtr> CreateDiffActFuncLookUpTable();
static const map<string, FnPtr> CreateOutputFuncLookUpTable();
static const map<string, ErrFuncPtr> CreateErrFuncLookUpTable();
static const map<string, DiffErrFuncPtr> CreateDiffErrFuncLookUpTable();

static const map<string, FnPtr> act_func_Look_up_table = CreateActFuncLookUpTable();
static const map<string, FnPtr> diff_act_func_look_up_table = CreateDiffActFuncLookUpTable();
static const map<string, FnPtr> output_func_Look_up_table = CreateOutputFuncLookUpTable();
static const map<string, ErrFuncPtr> err_func_Look_up_table = CreateErrFuncLookUpTable();
static const map<string, DiffErrFuncPtr> diff_err_func_look_up_table =CreateDiffErrFuncLookUpTable();

// Activation functions for hidden layer
static inline mat Sigmoid(mat const &pre_act) {
  return 1/(1 + exp(-pre_act));
}

static inline mat DiffSigmoid(mat const &act) {
  return act * (1-act);
}

static inline mat Tanh(mat const &pre_act) {
  return (exp(pre_act) - exp(-pre_act)) / (exp(pre_act) + exp(-pre_act));
}

static inline mat DiffTanh(mat const &act) {
  return 1-pow(act,2);
}
static mat inline Identity(mat const &pre_act) {
  return pre_act;
}

static mat inline DiffIdentity(mat const &act) {
  return act;
}

// Activation functions for output layer
static mat inline Softmax(mat const &pre_act) {
  mat temp = exp(pre_act);
  colvec denom = sum(temp, 1);
  temp.each_col() /= denom;
  return temp;
}

// Error function used for Neural Net
static double inline CrossEntropy(mat const &pred, mat const &y) {
  mat ln_pred = log(pred);
  double result = -sum(sum(ln_pred % y));
  return result;
}

static mat inline DiffCrossEntropy(mat const &pred, mat const &y, ActFunction const &act_func) {
  return pred - y;
}

static const map<string, FnPtr> CreateActFuncLookUpTable() {
  map<string, FnPtr> myMap;
  myMap["sigmoid"] = Sigmoid;
  myMap["tanh"] = Tanh;
  myMap["identity"] = Identity;
  return myMap;
}

static const map<string, FnPtr> CreateDiffActFuncLookUpTable() {
  map<string, FnPtr> myMap;
  myMap["sigmoid"] = DiffSigmoid;
  myMap["tanh"] = DiffTanh;
  myMap["identity"] = DiffIdentity;
  return myMap;
}


static const map<string, FnPtr> CreateOutputFuncLookUpTable() {
  map<string, FnPtr> myMap;
  myMap["softmax"] = Softmax;
  return myMap;
}

static const map<string, ErrFuncPtr> CreateErrFuncLookUpTable() {
  map<string, ErrFuncPtr> myMap;
  myMap["crossentropy"] = CrossEntropy;
  return myMap;
}

static const map<string, DiffErrFuncPtr> CreateDiffErrFuncLookUpTable() {
  map<string, DiffErrFuncPtr> myMap;
  myMap["crossentropy"] = DiffCrossEntropy;
  return myMap;
}

inline mat DComputeActFunc(mat const &pre_act, string act_func) {
  mat act = act_func_Look_up_table.at(act_func)(pre_act);
  return act;
}

inline mat DDiffActFunc(mat const &act, string act_func) {
  mat diff = diff_act_func_look_up_table.at(act_func)(act);
  return diff;
}

inline mat DComputeOutputFunc(mat const &pre_act, string output_func) {
  mat act = output_func_Look_up_table.at(output_func)(pre_act);
  return act;
}

inline mat DDiffOutputFunc(mat const &pre_act, string output_func) {
  cout << "No default differentiation of output function, return pre-activation function" << endl;
  return pre_act;
}

inline double DComputeErrFunc(mat const &y, mat const &pred, string err_func) {
  double result = err_func_Look_up_table.at(err_func)(pred, y);
  return result;
}

inline mat DDiffErrFunc(mat const &pred, mat const &y, string const errfunc, ActFunction const &act_func) {
  mat result = diff_err_func_look_up_table.at(errfunc)(pred, y, act_func);
  return result;
}

inline mat GradientDescent(double alpha, mat const &old_w, mat const &grad) { return old_w - alpha * grad; }

#endif /* ActFuncOp_hpp */
