/* file: neural_networks_prediction.cpp */
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

//++
//  Implementation of neural networks calculation functions.
//--

#include "neural_networks_prediction_result.h"
#include "neural_networks_prediction_model.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_PREDICTION_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_NEURAL_NETWORKS_PREDICTION_MODEL_ID);
}
}
}
}
}
