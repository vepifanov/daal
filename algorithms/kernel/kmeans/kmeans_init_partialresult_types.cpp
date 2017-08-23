/* file: kmeans_init_partialresult_types.cpp */
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
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_init_types.h"
#include "daal_defines.h"
#include "kmeans_init_impl.h"
#include "memory_block.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_KMEANS_INIT_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedStep2LocalPlusPlusPartialResult, SERIALIZATION_KMEANS_INIT_STEP2LOCAL_PP_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedStep3MasterPlusPlusPartialResult, SERIALIZATION_KMEANS_INIT_STEP3MASTER_PP_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedStep4LocalPlusPlusPartialResult, SERIALIZATION_KMEANS_INIT_STEP4LOCAL_PP_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedStep5MasterPlusPlusPartialResult, SERIALIZATION_KMEANS_INIT_STEP5MASTER_PP_PARTIAL_RESULT_ID);

PartialResult::PartialResult() : daal::algorithms::PartialResult(2) {}

/**
 * Returns a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] id   Identifier of the partial result
 * \return         Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the object
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the number of features in the result table of the K-Means algorithm
* \return Number of features in the result table of the K-Means algorithm
*/
size_t PartialResult::getNumberOfFeatures() const
{
    NumericTablePtr clusters = get(partialClusters);
    return clusters->getNumberOfColumns();
}

#define isPlusPlusMethod(method)\
    ((method == kmeans::init::plusPlusDense) || (method == kmeans::init::plusPlusCSR) || \
    (method == kmeans::init::parallelPlusDense) || (method == kmeans::init::parallelPlusCSR))

/**
* Checks a partial result of computing initial clusters for the K-Means algorithm
* \param[in] input   %Input object for the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
void PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    size_t inputFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    const size_t nClusters = (isPlusPlusMethod(method) ? 1 : kmPar->nClusters);

    int unexpectedLayouts = (int)packed_mask;

    NumericTablePtr pPartialClusters = get(partialClusters);
    if(pPartialClusters.get() && !checkNumericTable(pPartialClusters.get(), this->_errors.get(), partialClustersStr(),
        unexpectedLayouts, 0, inputFeatures, nClusters)) { return; }
    if (!checkNumericTable(get(partialClustersNumber).get(), this->_errors.get(), partialClustersNumberStr(),
        unexpectedLayouts, 0, 1, 1)) { return; }

    if(dynamic_cast<const Input*>(input))
    {
        DAAL_CHECK_EX(kmPar->nRowsTotal > 0, ErrorIncorrectParameter, ParameterName, nRowsTotalStr());
        DAAL_CHECK_EX(kmPar->nRowsTotal != kmPar->offset, ErrorIncorrectParameter, ParameterName, offsetStr());
    }
}

/**
 * Checks a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method of the algorithm
 */
void PartialResult::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    int unexpectedLayouts = (int)packed_mask;
    const size_t nClusters = (isPlusPlusMethod(method) ? 1 : kmPar->nClusters);

    if(!checkNumericTable(get(partialClustersNumber).get(), this->_errors.get(), partialClustersNumberStr(),
        unexpectedLayouts, 0, 1, 1)) {
        return;
    }
    NumericTablePtr pPartialClusters = get(partialClusters);
    if(pPartialClusters.get())
    {
        const size_t nRows = pPartialClusters->getNumberOfRows();
        DAAL_CHECK_EX(nRows <= kmPar->nClusters, ErrorIncorrectNumberOfRows, ArgumentName, partialClustersStr());
        if(checkNumericTable(pPartialClusters.get(), this->_errors.get(), partialClustersStr(), unexpectedLayouts))
            return;
    }
}
DistributedStep2LocalPlusPlusPartialResult::DistributedStep2LocalPlusPlusPartialResult() : daal::algorithms::PartialResult(3) {}

data_management::NumericTablePtr DistributedStep2LocalPlusPlusPartialResult::get(DistributedStep2LocalPlusPlusPartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedStep2LocalPlusPlusPartialResult::set(DistributedStep2LocalPlusPlusPartialResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

data_management::DataCollectionPtr DistributedStep2LocalPlusPlusPartialResult::get(DistributedStep2LocalPlusPlusPartialResultDataId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedStep2LocalPlusPlusPartialResult::set(DistributedStep2LocalPlusPlusPartialResultDataId id, const data_management::DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

void DistributedStep2LocalPlusPlusPartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const DistributedStep2LocalPlusPlusParameter *kmPar = static_cast<const DistributedStep2LocalPlusPlusParameter *>(par);

    if(!checkNumericTable(get(outputOfStep2ForStep3).get(), this->_errors.get(), outputOfStep2ForStep3Str(), (int)packed_mask, 0, 1, 1))
        return;

    if(kmPar->firstIteration)
        internal::checkLocalData(get(internalResult).get(), kmPar, internalResultStr(),
            static_cast<const Input *>(input)->get(data).get(), isParallelPlusMethod(method), this->_errors.get());
}

void DistributedStep2LocalPlusPlusPartialResult::check(const daal::algorithms::Parameter *par, int method) const
{
    //TODO
}

void DistributedStep2LocalPlusPlusPartialResult::initialize(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method)
{
    const DistributedStep2LocalPlusPlusParameter *kmPar = static_cast<const DistributedStep2LocalPlusPlusParameter *>(par);
    if(kmPar->firstIteration)
    {
        const auto pLocalData = get(internalResult);
        const auto pNClusters = NumericTable::cast(pLocalData->get(internal::numberOfClusters));
        if (!pNClusters) return;
        BlockDescriptor<int> block;
        pNClusters->getBlockOfRows(0, 1, writeOnly, block);
        *block.getBlockPtr() = 0;
        pNClusters->releaseBlockOfRows(block);
    }
}

DistributedStep3MasterPlusPlusPartialResult::DistributedStep3MasterPlusPlusPartialResult() : daal::algorithms::PartialResult(2)
{
    set(outputOfStep3ForStep4, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
    set(rngState, SerializationIfacePtr(new MemoryBlock()));
}

data_management::KeyValueDataCollectionPtr DistributedStep3MasterPlusPlusPartialResult::get(DistributedStep3MasterPlusPlusPartialResultId id) const
{
    return data_management::KeyValueDataCollection::cast(Argument::get(id));
}

data_management::NumericTablePtr DistributedStep3MasterPlusPlusPartialResult::get(DistributedStep3MasterPlusPlusPartialResultId id, size_t key) const
{
    data_management::KeyValueDataCollectionPtr pColl = data_management::KeyValueDataCollection::cast(Argument::get(outputOfStep3ForStep4));
    return data_management::NumericTable::cast((*pColl)[key]);
}

data_management::SerializationIfacePtr DistributedStep3MasterPlusPlusPartialResult::get(DistributedStep3MasterPlusPlusPartialResultDataId id) const
{
    return Argument::get(id);
}

void DistributedStep3MasterPlusPlusPartialResult::add(DistributedStep3MasterPlusPlusPartialResultId id, size_t key,
    const data_management::NumericTablePtr &ptr)
{
    data_management::KeyValueDataCollectionPtr pColl = data_management::KeyValueDataCollection::cast(Argument::get(outputOfStep3ForStep4));
    (*pColl)[key] = ptr;
}

void DistributedStep3MasterPlusPlusPartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    this->check(par, method);
}

void DistributedStep3MasterPlusPlusPartialResult::check(const daal::algorithms::Parameter *par, int method) const
{
    auto pArg = Argument::get(outputOfStep3ForStep4);
    DAAL_CHECK_EX(pArg.get(), ErrorNullPartialResult, ArgumentName, outputOfStep3ForStep4Str());
    data_management::KeyValueDataCollectionPtr pColl = data_management::KeyValueDataCollection::cast(pArg);
    DAAL_CHECK_EX(pColl.get(), ErrorNullInputDataCollection, ArgumentName, outputOfStep3ForStep4Str());
    pArg = Argument::get(rngState);
    DAAL_CHECK_EX(pArg.get(), ErrorNullPartialResult, ArgumentName, rngStateStr());
    data_management::MemoryBlockPtr pMemBlock = data_management::MemoryBlock::cast(pArg);
    DAAL_CHECK_EX(pMemBlock.get(), ErrorIncorrectItemInDataCollection, ArgumentName, rngStateStr());
}

void DistributedStep3MasterPlusPlusPartialResult::initialize(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method)
{
    auto pColl = get(outputOfStep3ForStep4);
    pColl->clear();
}

DistributedStep4LocalPlusPlusPartialResult::DistributedStep4LocalPlusPlusPartialResult() : daal::algorithms::PartialResult(1) {}

data_management::NumericTablePtr DistributedStep4LocalPlusPlusPartialResult::get(DistributedStep4LocalPlusPlusPartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedStep4LocalPlusPlusPartialResult::set(DistributedStep4LocalPlusPlusPartialResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

void DistributedStep4LocalPlusPlusPartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const DistributedStep4LocalPlusPlusInput* kmInput = static_cast<const DistributedStep4LocalPlusPlusInput*>(input);
    const auto nFeatures = kmInput->get(data)->getNumberOfColumns();
    data_management::NumericTablePtr pInput = kmInput->get(inputOfStep4FromStep3);
    const auto nRows = pInput->getNumberOfColumns();

    if(!checkNumericTable(get(outputOfStep4).get(), this->_errors.get(), outputOfStep4Str(),
        (int)packed_mask, 0, nFeatures, nRows))
        return;
}

void DistributedStep4LocalPlusPlusPartialResult::check(const daal::algorithms::Parameter *par, int method) const
{
    //TODO??
}

DistributedStep5MasterPlusPlusPartialResult::DistributedStep5MasterPlusPlusPartialResult() : daal::algorithms::PartialResult(2) {}

data_management::NumericTablePtr DistributedStep5MasterPlusPlusPartialResult::get(DistributedStep5MasterPlusPlusPartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedStep5MasterPlusPlusPartialResult::set(DistributedStep5MasterPlusPlusPartialResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

void DistributedStep5MasterPlusPlusPartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *stepPar = static_cast<const Parameter *>(par);
    const DistributedStep5MasterPlusPlusInput* kmInput = static_cast<const DistributedStep5MasterPlusPlusInput*>(input);

    const size_t nMaxCandidates = size_t(stepPar->oversamplingFactor*stepPar->nClusters)*stepPar->nRounds + 1;
    const auto pColl = kmInput->get(inputCentroids);
    data_management::NumericTablePtr pCentroids = data_management::NumericTable::cast((*pColl)[0]);
    const auto nFeatures = pCentroids->getNumberOfColumns();

    if(!checkNumericTable(get(candidates).get(), this->_errors.get(), candidatesStr(),
        (int)packed_mask, 0, nFeatures, nMaxCandidates))
        return;

    if(!checkNumericTable(get(weights).get(), this->_errors.get(), candidateRatingStr(),
        (int)packed_mask, 0, nMaxCandidates, 1))
        return;
}

void DistributedStep5MasterPlusPlusPartialResult::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *stepPar = static_cast<const Parameter *>(par);
    const size_t nMaxCandidates = size_t(stepPar->oversamplingFactor*stepPar->nClusters)*stepPar->nRounds + 1;

    if(!checkNumericTable(get(candidates).get(), this->_errors.get(), candidatesStr(),
        (int)packed_mask, 0, 0, nMaxCandidates))
        return;

    if(!checkNumericTable(get(weights).get(), this->_errors.get(), candidateRatingStr(),
        (int)packed_mask, 0, nMaxCandidates, 1))
        return;
}

} // namespace interface1
} // namespace init
} // namespace kmeans
} // namespace algorithm
} // namespace daal
