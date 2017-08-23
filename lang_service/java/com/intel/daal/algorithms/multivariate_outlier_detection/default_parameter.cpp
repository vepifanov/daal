/* file: default_parameter.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Parameter */

#include "daal.h"
#include "multivariate_outlier_detection/defaultdense/JParameter.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::multivariate_outlier_detection;

/*
 * Class:     com_intel_daal_algorithms_multivariate_1outlier_1detection_defaultdense_Parameter
 * Method:    cSetInitializationProcedure
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_defaultdense_Parameter_cSetInitializationProcedure
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong initAddr)
{
    using namespace daal::algorithms;
    multivariate_outlier_detection::Parameter<defaultDense> *parameterAddr = (multivariate_outlier_detection::Parameter<defaultDense> *)parAddr;
    parameterAddr->initializationProcedure = *(services::SharedPtr<multivariate_outlier_detection::InitIface> *)initAddr;
}
