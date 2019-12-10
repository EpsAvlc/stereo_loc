/*
 * Created on Tue Dec 10 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */
#include <iostream>

namespace math_tools
{
    class GuassainDistribution
    {
    public:
        GuassainDistribution(float mu, float sigma) : mu_(mu), sigma_(sigma){};
        float CalcProbability(float x);
    private:
        float mu_, sigma_;
    };
};
