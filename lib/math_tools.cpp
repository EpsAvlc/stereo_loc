/*
 * Created on Tue Dec 10 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "math_tools.h"

#include <cmath>
using namespace math_tools;

float GuassainDistribution::CalcProbability(float x)
{
    return (1 / (sigma_ * sqrt(2*M_PI))) * exp(-(x-mu_)*(x-mu_)/(2 * sigma_ * sigma_));
}
