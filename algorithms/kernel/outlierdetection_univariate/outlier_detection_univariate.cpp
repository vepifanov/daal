/* file: outlier_detection_univariate.cpp */
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
//  Outlier Detection algorithm parameter structure
//--
*/

#include "outlier_detection_univariate_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_OUTLIER_DETECTION_UNIVARIATE_RESULT_ID);

/**
* Returns the initial value for the univariate outlier detection algorithm
* \param[in] data          Pointer to input values of size n x p
* \param[in] location      Vector of mean estimates of size 1 x p
* \param[in] scatter       Measure of spread, the array of standard deviations of size 1 x p
* \param[in] threshold     Limit that defines the outlier region, the array of non-negative numbers of size 1 x p
*/
void DefaultInit::operator()(data_management::NumericTable *data,
                            data_management::NumericTable *location,
                            data_management::NumericTable *scatter,
                            data_management::NumericTable *threshold)
{
    size_t nFeatures = data->getNumberOfColumns();

    BlockDescriptor<double> locationBlock;
    location->getBlockOfRows(0, 1, writeOnly, locationBlock);
    double *locationArray = locationBlock.getBlockPtr();
    BlockDescriptor<double> scatterBlock;
    scatter->getBlockOfRows(0, 1, writeOnly, scatterBlock);
    double *scatterArray = scatterBlock.getBlockPtr();
    BlockDescriptor<double> thresholdBlock;
    threshold->getBlockOfRows(0, 1, writeOnly, thresholdBlock);
    double *thresholdArray = thresholdBlock.getBlockPtr();
    for(size_t i = 0; i < nFeatures; i++)
    {
        locationArray[i]  = 0.0;
        scatterArray[i]   = 1.0;
        thresholdArray[i] = 3.0;
    }

    location->releaseBlockOfRows(locationBlock);
    scatter->releaseBlockOfRows(scatterBlock);
    threshold->releaseBlockOfRows(thresholdBlock);
}

Parameter::Parameter() : daal::algorithms::Parameter(), initializationProcedure(new DefaultInit()) {}

void Parameter::check() const {}

Input::Input() : daal::algorithms::Input(1) {}

/**
 * Returns an input object for the univariate outlier detection algorithm
 * \param[in] id    Identifier of the %input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the univariate outlier detection algorithm
 * \param[in] id    Identifier of the %input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks input objects for the univariate outlier detection algorithm
 * \param[in] par     Parameters of the algorithm
 * \param[in] method  univariate outlier detection computation method
      */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if (!checkNumericTable(get(data).get(), this->_errors.get(), dataStr())) { return; }
}

Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns a result of the univariate outlier detection algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets a result of the univariate outlier detection algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result object of the univariate outlier detection algorithm
 * \param[in] input   Pointer to the  %input objects for the algorithm
 * \param[in] par     Pointer to the parameters of the algorithm
 * \param[in] method  univariate outlier detection computation method
 * \return             Status of checking
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nFeatures = algInput->get(data)->getNumberOfColumns();
    size_t nVectors  = algInput->get(data)->getNumberOfRows();

    int unexpectedLayouts = packed_mask;
    if (!checkNumericTable(get(weights).get(), this->_errors.get(), weightsStr(), unexpectedLayouts, 0, nFeatures, nVectors)) { return; }
}

} // namespace interface1
} // namespace univariate_outlier_detection
} // namespace algorithms
} // namespace daal
