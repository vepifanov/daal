/* file: truncated_gaussian_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of truncated gaussian initializer
//--
*/
#ifndef __TRUNCATED_GAUSSIAN_INITIALIZER_IMPL_I__
#define __TRUNCATED_GAUSSIAN_INITIALIZER_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{
namespace internal
{
/* result = sigma * [ ICDF( CDF(a) + uniform(0,1) * ( CDF(b) - CDF(a)) ) ] + mean;
    CDF(a) = 0, if a = -inf;
    CDF(b) = 1, if b = +inf;
*/
template<typename algorithmFPType, Method method, CpuType cpu>
void TruncatedGaussianKernel<algorithmFPType, method, cpu>::compute(const truncated_gaussian::Parameter<algorithmFPType> *parameter, Tensor *resultTensor)
{
    BaseRNGs<cpu> baseRng(parameter->seed);
    RNGs<algorithmFPType, cpu> rng;

    size_t size = resultTensor->getSize();

    algorithmFPType mean     = (algorithmFPType)parameter->mean;
    algorithmFPType sigma    = (algorithmFPType)parameter->sigma;
    algorithmFPType cdf_b    = getCDFNormal(parameter->b, mean, sigma);
    algorithmFPType cdf_a    = getCDFNormal(parameter->a, mean, sigma);
    algorithmFPType cdf_diff = cdf_b - cdf_a;

    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(resultTensor, 0, 0, 0, resultTensor->getDimensions()[0]);
    algorithmFPType *resultArray = resultSubtensor.get();

    rng.uniform(size, resultArray, baseRng, 0.0, 1.0);

    size_t nBlocks = size / _nElemsInBlock;
    nBlocks += (nBlocks * _nElemsInBlock != size);

    daal::threader_for(nBlocks, nBlocks, [ & ](int block)
    {
        size_t nElemsToProcess = _nElemsInBlock;
        size_t shift = block * _nElemsInBlock;

        algorithmFPType *resultLocal = &resultArray[shift];

        if( block == nBlocks - 1 )
        {
            nElemsToProcess = size - shift;
        }

        for(size_t i = 0; i < nElemsToProcess; i++)
        {
            resultLocal[i] = cdf_a + resultLocal[i] * cdf_diff;
        }

        Math<algorithmFPType,cpu>::vCdfNormInv(nElemsToProcess, resultLocal, resultLocal);

        for(size_t i = 0; i < nElemsToProcess; i++)
        {
            resultLocal[i] = sigma * resultLocal[i] + mean;
        }
    } );
}

template<typename algorithmFPType, Method method, CpuType cpu>
algorithmFPType TruncatedGaussianKernel<algorithmFPType, method, cpu>::getCDFNormal(algorithmFPType p, algorithmFPType mean, algorithmFPType sigma)
{
    return (algorithmFPType)0.5 * ( (algorithmFPType)1.0 + Math<algorithmFPType, cpu>::sErf( (p - mean) / (sigma * sqrt(2) )) );
}

} // internal
} // namespace truncated_gaussian
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
