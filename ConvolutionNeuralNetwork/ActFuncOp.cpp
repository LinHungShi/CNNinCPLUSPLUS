//
//  ActFuncOp.cpp
//  ConvolutionNeuralNetwork
//
//  Created by Lin Hung-Shi on 2/16/16.
//  Copyright © 2016 Lin Hung-Shi. All rights reserved.
//

#include "ActFuncOp.hpp"

using namespace arma;
using namespace std;

typedef mat (*FnPtr)(mat input);
typedef double (*ErrFuncPtr)(mat pred, mat y);
typedef mat (*DiffErrFuncPtr)(mat pred, mat y);

static const map<string, FnPtr> CreateActFuncLookUpTable();
static const map<string, FnPtr> CreateDiffActFuncLookUpTable();
static const map<string, FnPtr> CreateOutputFuncLookUpTable();
static const map<string, ErrFuncPtr> CreateErrFuncLookUpTable();
static const map<string, DiffErrFuncPtr> CreateDiffErrFuncLookUpTable();

static mat Sigmoid(mat const);
static mat Tanh(mat const);
static mat Identity(mat const);
static mat DiffSigmoid(mat const);
static mat DiffTanh(mat const);
static mat DiffIdentity(mat const);
static mat Softmax(mat const);
static mat DiffCrossEntropy(mat const, mat const);
static double CrossEntropy(mat const, mat const);

static const map<string, FnPtr> act_func_Look_up_table = CreateActFuncLookUpTable();
static const map<string, FnPtr> diff_act_func_look_up_table = CreateDiffActFuncLookUpTable();
static const map<string, FnPtr> output_func_Look_up_table = CreateOutputFuncLookUpTable();
static const map<string, ErrFuncPtr> err_func_Look_up_table = CreateErrFuncLookUpTable();
static const map<string, DiffErrFuncPtr> diff_err_func_look_up_table = CreateDiffErrFuncLookUpTable();



static mat Sigmoid(mat const pre_act){
    
    return 1/(1 + exp(-pre_act));
}

static mat DiffSigmoid(mat const act){
    
    return act * (1-act);
}

static mat Tanh(mat const pre_act){
    
    return (exp(pre_act) - exp(-pre_act)) / (exp(pre_act) + exp(-pre_act));
}

static mat DiffTanh(mat const act){
    
    return 1-pow(act,2);
}
static mat Identity(mat const pre_act){
    
    return pre_act;
}

static mat DiffIdentity(mat const act){
    
    return act;
}

static mat Softmax(mat const pre_act){
    
    mat temp = exp(pre_act);
    colvec denom = sum(temp, 1);
    temp.each_col() /= denom;
    return temp;
}


static double CrossEntropy(mat pred, mat y){
    
    mat ln_pred = log(pred);
    double result = -sum(sum(ln_pred % y));
    return result;
}

static mat DiffCrossEntropy(mat pred, mat y){
    
    return pred - y;
}

static const map<string, FnPtr> CreateActFuncLookUpTable(){
    
    map<string, FnPtr> myMap;
    myMap["sigmoid"] = Sigmoid;
    myMap["tanh"] = Tanh;
    myMap["identity"] = Identity;
    return myMap;
}

static const map<string, FnPtr> CreateDiffActFuncLookUpTable(){
    
    map<string, FnPtr> myMap;
    myMap["sigmoid"] = DiffSigmoid;
    myMap["tanh"] = DiffTanh;
    myMap["identity"] = DiffIdentity;
    return myMap;
}


static const map<string, FnPtr> CreateOutputFuncLookUpTable(){
    
    map<string, FnPtr> myMap;
    myMap["softmax"] = Softmax;
    
    return myMap;
}

static const map<string, ErrFuncPtr> CreateErrFuncLookUpTable(){
    
    map<string, ErrFuncPtr> myMap;
    myMap["crossentropy"] = CrossEntropy;
    return myMap;
}

static const map<string, DiffErrFuncPtr> CreateDiffErrFuncLookUpTable(){
    
    map<string, DiffErrFuncPtr> myMap;
    myMap["crossentropy"] = DiffCrossEntropy;
    return myMap;
}

mat DComputeActFunc(mat pre_act, string act_func){
    
    mat act = act_func_Look_up_table.at(act_func)(pre_act);
    return act;
}

mat DDiffActFunc(mat act, string act_func){
    
    mat diff = diff_act_func_look_up_table.at(act_func)(act);
    return diff;
}

mat DComputeOutputFunc(mat pre_act, string output_func){
    
    mat act = output_func_Look_up_table.at(output_func)(pre_act);
    return act;
}

mat DDiffOutputFunc(mat pre_act, string output_func)
{
    cout << "No default differentiation of output function, return pre-activation function" << endl;
    return pre_act;
    
}

double DComputeErrFunc(mat y, mat pred, string err_func){
    
    double result = err_func_Look_up_table.at(err_func)(pred, y);
    return result;
}

mat DDiffErrFunc(mat pred, mat y, string errfunc){

    mat result = diff_err_func_look_up_table.at(errfunc)(pred, y);
    return result;
}



