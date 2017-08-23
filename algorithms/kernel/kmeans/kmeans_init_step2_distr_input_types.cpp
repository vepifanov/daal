/* file: kmeans_init_step2_distr_input_types.cpp */
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

DistributedStep2MasterInput::DistributedStep2MasterInput() : InputIface(1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}

/**
* Returns an input object for computing initial clusters for the K-Means algorithm
* in the second step of the distributed processing mode
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
DataCollectionPtr DistributedStep2MasterInput::get(DistributedStep2MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
* Sets an input object for computing initial clusters for the K-Means algorithm
* in the second step of the distributed processing mode
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void DistributedStep2MasterInput::set(DistributedStep2MasterInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, staticPointerCast<SerializationIface, DataCollection>(ptr));
}

/**
 * Adds a value to the data collection of input objects for computing initial clusters for the K-Means algorithm
 * in the second step of the distributed processing mode
 * \param[in] id    Identifier of the parameter
 * \param[in] value Pointer to the new parameter value
 */
void DistributedStep2MasterInput::add(DistributedStep2MasterInputId id, const SharedPtr<PartialResult> &value)
{
    DataCollectionPtr collection
        = staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
    collection->push_back( value );
}

/**
* Returns the number of features in the Input data table in the second step of the distributed processing mode
* \return Number of features in the Input data table
*/
size_t DistributedStep2MasterInput::getNumberOfFeatures() const
{
    DataCollectionPtr collection = get(partialResults);
    SharedPtr<PartialResult> pres = staticPointerCast<PartialResult, SerializationIface>((*collection)[0]);
    return pres->getNumberOfFeatures();
}

template <typename Type>
Type getSingleValue(NumericTable& tbl)
{
    BlockDescriptor<Type> block;
    tbl.getBlockOfRows(0, 1, readOnly, block);
    Type value = block.getBlockPtr()[0];
    tbl.releaseBlockOfRows(block);
    return value;
}

static bool checkPartialResult(const SerializationIfacePtr& ptr, const Parameter *kmPar, services::ErrorCollection *errors,
    int unexpectedLayouts, size_t nFeatures, int& nClusters)
{
    SharedPtr<PartialResult> pPres = dynamicPointerCast<PartialResult, SerializationIface>(ptr);
    if(!pPres)
    {
        errors->add(ErrorIncorrectElementInPartialResultCollection);
        return false;
    }

    auto pPartialClustersNumber = pPres->get(partialClustersNumber);
    if(!checkNumericTable(pPartialClustersNumber.get(), errors, partialClustersNumberStr(), unexpectedLayouts, 0, 1, 1))
        return false;

    int nPartClusters = getSingleValue<int>(*pPartialClustersNumber);
    if((nPartClusters < 0) || (nPartClusters > kmPar->nClusters))
    {
        errors->add(ErrorIncorrectNumberOfPartialClusters);
        return false;
    }
    auto pPartialClusters = pPres->get(partialClusters);
    if(pPartialClusters.get())
    {
        if(pPartialClusters->getNumberOfRows() < nPartClusters)
        {
            errors->add(ErrorIncorrectNumberOfPartialClusters);
            return false;
        }
        if(!checkNumericTable(pPartialClusters.get(), errors, partialClustersStr(),
            unexpectedLayouts, 0, nFeatures, pPartialClusters->getNumberOfRows()))
            return false;
        nClusters += nPartClusters;
    }
    else if(nPartClusters)
    {
        errors->add(ErrorIncorrectNumberOfPartialClusters);
        return false;
    }
    return true;
}

/**
* Checks an input object for computing initial clusters for the K-Means algorithm
* in the second step of the distributed processing mode
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
void DistributedStep2MasterInput::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *kmPar = static_cast<const Parameter *>(par);
    DataCollectionPtr collection = get(partialResults);

    DAAL_CHECK(collection, ErrorNullInputDataCollection);
    const size_t nBlocks = collection->size();
    DAAL_CHECK(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables);

    const int unexpectedLayouts = (int)packed_mask;
    int nClusters = 0;
    int nFeatures = 0;
    if(!checkPartialResult((*collection)[0], kmPar, this->_errors.get(), unexpectedLayouts, nFeatures, nClusters))
        return;

    SharedPtr<PartialResult> firstPres = dynamicPointerCast<PartialResult, SerializationIface>((*collection)[0]);
    nFeatures = firstPres->get(partialClusters)->getNumberOfColumns();
    for(size_t i = 1; i < nBlocks; i++)
    {
        if(!checkPartialResult((*collection)[i], kmPar, this->_errors.get(), unexpectedLayouts, nFeatures, nClusters))
            return;
    }
    DAAL_CHECK(nClusters == kmPar->nClusters, ErrorIncorrectTotalNumberOfPartialClusters);
}

} // namespace interface1
} // namespace init
} // namespace kmeans
} // namespace algorithm
} // namespace daal
