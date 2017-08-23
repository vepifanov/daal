/* file: kmeans_init_impl.i */
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
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "algorithm.h"
#include "numeric_table.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_rng.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{

template <Method method, typename algorithmFPType, CpuType cpu>
bool init( size_t p, size_t n, size_t nRowsTotal, size_t nClusters, algorithmFPType *clusters,
    NumericTable *ntData, unsigned int seed, size_t& clustersFound, services::KernelErrorCollection *errors)
{
    if(method == deterministicDense || method == deterministicCSR)
    {
        BlockMicroTable<algorithmFPType, readOnly, cpu> mtData(ntData);
        algorithmFPType *data;
        mtData.getBlockOfRows(0, nClusters, &data);

        for (size_t i = 0; i < nClusters && i < n; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                clusters[i * p + j] = data[i * p + j];
            }
        }

        mtData.release();

        clustersFound = nClusters;
    }
    else if(method == randomDense || method == randomCSR)
    {
        int *indices = (int *)daal::services::daal_malloc( sizeof(int) * nClusters );
        if( !indices )
        {
            return false;
        }

        BlockMicroTable<algorithmFPType, readOnly, cpu> mtData(ntData);
        algorithmFPType *data;

        BaseRNGs<cpu> baseRng(seed);
        RNGs<int, cpu> rng;

        size_t k = 0;
        for(size_t i = 0; i < nClusters; i++)
        {
            int errCode = rng.uniform(1, &indices[i], baseRng, i, (int)nRowsTotal);
            if(errCode) { errors->add(ErrorIncorrectErrorcodeFromGenerator); }

            size_t c = (size_t)indices[i];

            int value = indices[i];
            for(size_t j = i; j > 0; j--)
            {
                if(value == indices[j-1])
                {
                    c = (size_t)(j-1);
                    value = c;
                }
            }

            if(c>=n )
                continue;

            mtData.getBlockOfRows( c, 1, &data );

            for(size_t j = 0; j < p; j++)
            {
                clusters[k * p + j] = data[j];
            }
            k++;

            mtData.release();
        }
        clustersFound = k;

        daal::services::daal_free( indices );
    }

    return true;
}

template <Method method, typename algorithmFPType, CpuType cpu>
void KMeansinitKernel<method, algorithmFPType, cpu>::compute( size_t na, const NumericTable *const *a,
                                                     size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData     = const_cast<NumericTable *>( a[0] );
    NumericTable *ntClusters = const_cast<NumericTable *>( r[0] );

    const size_t p = ntData->getNumberOfColumns();
    const size_t n = ntData->getNumberOfRows();
    const size_t nClusters = par->nClusters;

    WriteOnlyRows<algorithmFPType, cpu> clustersBD(ntClusters, 0, nClusters);
    algorithmFPType *clusters = clustersBD.get();

    size_t clustersFound = 0;
    if( !init<method, algorithmFPType, cpu>( p, n, n, nClusters, clusters, ntData, par->seed, clustersFound, this->_errors.get()) )
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed); return;
    }
}

template <typename algorithmFPType, CpuType cpu>
bool initDistrDeterministic(const NumericTable* pData, const Parameter *par, size_t& nClustersFound, NumericTablePtr& pRes)
{
    nClustersFound = 0;
    if(par->nClusters <= par->offset)
        return true; //ok

    nClustersFound = par->nClusters - par->offset;
    const size_t nRows = pData->getNumberOfRows();
    if(nClustersFound > nRows)
        nClustersFound = nRows;
    if(!pRes)
    {
        pRes.reset(new HomogenNumericTableCPU<algorithmFPType, cpu>(pData->getNumberOfColumns(), nClustersFound));
        if(!pRes.get())
            return false;
    }

    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, nClustersFound);
    if(!resBD.get())
        return false;
    ReadRows<algorithmFPType, cpu> dataBD(*const_cast<NumericTable*>(pData), 0, nClustersFound);
    const size_t sz = pData->getNumberOfColumns()*nClustersFound*sizeof(algorithmFPType);
    daal::services::daal_memcpy_s(resBD.get(), sz, dataBD.get(), sz);
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool generateRandomIndices(const Parameter *par, size_t nRows, size_t& nClustersFound,
    int* clusters, services::KernelErrorCollection& errors)
{
    TArray<int, cpu> aIndices(par->nClusters);
    int* indices = aIndices.get();
    if(!indices)
        return false;
    BaseRNGs<cpu> baseRng(par->seed);
    RNGs<int, cpu> rng;
    nClustersFound = 0;
    for(size_t i = 0; i < par->nClusters; i++)
    {
        if(rng.uniform(1, &indices[i], baseRng, i, (int)par->nRowsTotal))
            errors.add(ErrorIncorrectErrorcodeFromGenerator);

        size_t c = (size_t)indices[i];
        int value = indices[i];
        for(size_t j = i; j > 0; j--)
        {
            if(value == indices[j - 1])
            {
                c = (size_t)(j - 1);
                value = c;
            }
        }

        if(c < par->offset || c >= par->offset + nRows)
            continue;
        clusters[nClustersFound] = c - par->offset;
        nClustersFound++;
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool initDistrRandom(const NumericTable* pData, const Parameter *par,
    size_t& nClustersFound, NumericTablePtr& pRes, services::KernelErrorCollection& errors)
{
    TArray<int, cpu> clusters(par->nClusters);
    if(!clusters.get() || !generateRandomIndices<algorithmFPType, cpu>(par, pData->getNumberOfRows(),
        nClustersFound, clusters.get(), errors))
        return false;

    if(!nClustersFound)
        return true;

    const size_t p = pData->getNumberOfColumns();
    if(!pRes)
    {
        pRes.reset(new HomogenNumericTableCPU<algorithmFPType, cpu>(p, nClustersFound));
        if(!pRes)
            return false;
    }

    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, nClustersFound);
    if(!resBD.get())
        return false;

    auto aClusters = resBD.get();
    ReadRows<algorithmFPType, cpu> dataBD;
    for(size_t i = 0; i < nClustersFound; ++i)
    {
        auto pRow = dataBD.set(const_cast<NumericTable*>(pData), clusters.get()[i], 1);
        for(size_t j = 0; j < p; j++)
            aClusters[i * p + j] = pRow[j];
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool initDistrPlusPlus(const NumericTable* pData, const Parameter *par,
    size_t& nClustersFound, NumericTablePtr& pRes)
{
    nClustersFound = 0;

    BaseRNGs<cpu> baseRng(par->seed);
    RNGs<int, cpu> rng;

    int index = 0;
    rng.uniform(1, &index, baseRng, 0, (int)par->nRowsTotal);
    size_t c(index);
    if(c < par->offset)
        return true; //ok
    if(c >= par->offset + pData->getNumberOfRows())
        return true; //ok

    ReadRows<algorithmFPType, cpu> dataBD(*const_cast<NumericTable*>(pData), c - par->offset, 1);
    if(!dataBD.get())
        return false;

    if(!pRes)
    {
        pRes.reset(new HomogenNumericTableCPU<algorithmFPType, cpu>(pData->getNumberOfColumns(), 1));
        if(!pRes)
            return false;
    }

    nClustersFound = 1;
    const size_t p = pData->getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, 1);
    daal::services::daal_memcpy_s(resBD.get(), sizeof(algorithmFPType)*p, dataBD.get(), sizeof(algorithmFPType)*p);
    return true;
}

template <Method method, typename algorithmFPType, CpuType cpu>
void KMeansinitStep1LocalKernel<method, algorithmFPType, cpu>::compute(const NumericTable* pData, const Parameter *par,
    NumericTable* pNumPartialClusters, NumericTablePtr& pPartialClusters)
{
    size_t nClustersFound = 0;
    if((((method == deterministicDense) || (method == deterministicCSR)) &&
        !initDistrDeterministic<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters)) ||
       (((method == randomDense) || (method == randomCSR)) &&
       !initDistrRandom<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters, *this->_errors)) ||
       (isPlusPlusMethod(method) &&
       !initDistrPlusPlus<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters)))
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }
    WriteOnlyRows<int, cpu> npcBD(*pNumPartialClusters, 0, 1);
    *npcBD.get() = (int)nClustersFound;
}

template <Method method, typename algorithmFPType, CpuType cpu>
void KMeansinitStep2MasterKernel<method, algorithmFPType, cpu>::finalizeCompute(size_t na, const NumericTable *const *a,
    NumericTable* ntClusters, const Parameter *par)
{
    size_t nBlocks = na / 2;
    size_t p = ntClusters->getNumberOfColumns();
    size_t nClusters = par->nClusters;

    BlockMicroTable<algorithmFPType, writeOnly, cpu> mtClusters( ntClusters );
    algorithmFPType *clusters;
    mtClusters.getBlockOfRows( 0, nClusters, &clusters );

    size_t k = 0;

    for( size_t i = 0; i<nBlocks; i++ )
    {
        if(!a[i * 2 + 1])
            continue; //can be null
        BlockMicroTable<int,    readOnly, cpu> mtInClustersN( a[i*2 + 0] );
        BlockMicroTable<algorithmFPType, readOnly, cpu> mtInClusters ( a[i*2 + 1] );

        int    *inClustersN;
        algorithmFPType *inClusters;

        mtInClusters.getBlockOfRows( 0, nClusters, &inClusters );
        mtInClustersN.getBlockOfRows( 0, 1, &inClustersN );

        size_t inK = *inClustersN;
        for( size_t j=0; j<inK; j++ )
        {
            for( size_t h=0; h<p; h++ )
            {
                clusters[k*p + h] = inClusters[j*p + h];
            }
            k++;
        }

        mtInClustersN.release();
        mtInClusters.release();
    }

    mtClusters.release();
}

} // namespace daal::algorithms::kmeans::init::internal
} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
