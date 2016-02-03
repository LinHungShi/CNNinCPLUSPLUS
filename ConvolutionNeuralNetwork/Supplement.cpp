//
//  Supplement.cpp
//  mytestproject
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "Supplement.hpp"

using namespace arma;
using namespace std;
mat actNeuron(mat pre_act, string actfun, string l_name){
    
    mat result;
    if(l_name == "Hid"){
        if(actfun == "sigmoid"){
            result = 1 / (1 + exp(-pre_act));
        }
        else if (actfun == "tanh"){
            result = (exp(pre_act) - exp(-pre_act)) / (exp(pre_act) + exp(-pre_act));
        }
    }
    else if(l_name == "Output"){
        if(actfun == "softmax"){
            mat temp = exp(pre_act);
            //cout << "temp:\n" << temp;
            colvec denom = sum(temp, 1);
            temp.each_col() /= denom;
            //cout <<"softmax:" << temp << endl;
            result = temp;
        }
    }
    return result;
}



mat diffAct(mat pre_act, string actfun){
    
    mat result;
    if(actfun == "sigmoid"){
        result = pre_act % (1 - pre_act);
    }
    else if (actfun == "tanh"){
        result = 1 - pow(pre_act, 2);
    }
    return result;
}

mat initWeight(int row, int col, string init_method){
    
    mat weight;
    if(init_method == "randn"){
        weight = mat(row, col, fill::randn);
    }
    return weight;
}

double computeError(mat y, mat predict){
    mat combin = y % log(predict);
    //cout << "y" << y <<endl;
    //cout << "predict" << log(predict) << endl;
    //cout << "combination" << combin << endl;
    //cout << sum(combin, 0) << endl;
    //cout << "max" << max(combin);
    return -accu(y % log(predict));
}

double errorRate(mat y, mat predict){
    colvec vec = predict.col(1);
    uvec num = find(vec>0.5);
    
    double rate = num.size() / vec.size();
    cout << "num_size:" << num.size() << endl;
    return rate;
}