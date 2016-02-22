//
//  Supplement.cpp
//  mytestproject
//
//  Created by Lin Hung-Shi on 11/23/15.
//  Copyright © 2015 Lin Hung-Shi. All rights reserved.
//

#include "Supplement.hpp"

void UpdateHidLayerParm(FullLayer &layer,
                     mat next_layer_delta,
                     mat next_layer_weight,
                     double alpha,
                     mat input)
{
    HidLayer &hid = static_cast<HidLayer&>(layer);
    hid.UpdateParm(alpha, next_layer_delta, next_layer_weight, input);
    
}

void UpdateOutputLayerParm(FullLayer &layer,
                           mat y,
                           mat input,
                           double alpha,
                           string err_func){
    
    OutputLayer &out = static_cast<OutputLayer&>(layer);
    out.UpdateParm(alpha, y, input, err_func);
    
}

