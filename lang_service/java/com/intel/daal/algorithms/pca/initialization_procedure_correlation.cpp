/* file: initialization_procedure_correlation.cpp */
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

#include <jni.h>

#include "daal.h"
#include "initialization_procedure.h"

#include "pca/JInitializationProcedureCorrelation.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::pca;

/*
 * Class:     com_intel_daal_algorithms_pca_InitializationProcedureCorrelation
 * Method:    cNewJavaInitializationProcedure
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_InitializationProcedureCorrelation_cNewJavaInitializationProcedure
(JNIEnv *env, jobject thisObj)
{
    JavaVM *jvm;

    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if(status != 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), "Couldn't get Java VM interface");
        return 0;
    }

    pca::PartialResultsInitIface<defaultDense> *initializationProcedure = new pca::JavaPartialResultInit<defaultDense>(jvm, thisObj);

    services::SharedPtr<pca::PartialResultsInitIface<defaultDense> > *initializationProcedureShPtr =
            new services::SharedPtr<pca::PartialResultsInitIface<defaultDense> >(initializationProcedure);

    return (jlong)initializationProcedureShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_pca_InitializationProcedureCorrelation
 * Method:    cInitIfaceDispose
 * Signature: (J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_InitializationProcedureCorrelation_cInitIfaceDispose
(JNIEnv *, jobject, jlong cInitIface)
{
    services::SharedPtr<pca::PartialResultsInitIface<defaultDense> > *ptr = (services::SharedPtr<pca::PartialResultsInitIface<defaultDense> > *) cInitIface;
    delete ptr;
}
