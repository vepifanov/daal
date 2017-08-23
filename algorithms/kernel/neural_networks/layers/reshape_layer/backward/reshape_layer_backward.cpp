/* file: reshape_layer_backward.cpp */
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
//  Implementation of reshape calculation algorithm and types methods.
//--
*/

#include "reshape_layer_backward_types.h"
#include "reshape_layer_types.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace reshape
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_RESHAPE_BACKWARD_RESULT_ID);
/** Default constructor */
Input::Input() {};

/**
* Returns an input object for the backward reshape layer
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets an input object for the backward reshape layer
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(LayerDataId id, const data_management::NumericTablePtr &value)
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    (*layerData)[id] = value;
}

/**
* Checks input object for the backward reshape layer
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);
    if (!parameter->propagateGradient) { return; }

    layers::backward::Input::check(par, method);
    if(this->_errors->size() > 0) { return; }
}

Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward reshape layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Input *in = static_cast<const Input *>(input);
    const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);

    if (!parameter->propagateGradient) { return; }

    data_management::NumericTablePtr dimsTable = in->get(layers::reshape::auxInputDimensions);

    size_t nDims = dimsTable->getNumberOfColumns();

    services::Collection<size_t> iDims( nDims );

    data_management::BlockDescriptor<int> block;
    dimsTable->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *dataArray = block.getBlockPtr();

    for(size_t i=0; i<nDims; i++)
    {
        iDims[i] = dataArray[i];
    }

    dimsTable->releaseBlockOfRows(block);

    if(!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), resultLayerDataStr(), &iDims)) { return; }
}

}// namespace interface1
}// namespace backward
}// namespace reshape
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
