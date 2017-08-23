/* file: initialization_procedure_svd.cpp */
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

#include "pca/JInitializationProcedureSVD.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::pca;

/*
 * Class:     com_intel_daal_algorithms_pca_InitializationProcedureSVD
 * Method:    cNewJavaInitializationProcedure
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_InitializationProcedureSVD_cNewJavaInitializationProcedure
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

    pca::PartialResultsInitIface<svdDense> *initializationProcedure = new pca::JavaPartialResultInit<svdDense>(jvm, thisObj);

    services::SharedPtr<pca::PartialResultsInitIface<svdDense> > *initializationProcedureShPtr =
        new services::SharedPtr<pca::PartialResultsInitIface<svdDense> >(initializationProcedure);

    return (jlong)initializationProcedureShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_pca_InitializationProcedureSVD
 * Method:    cInitIfaceDispose
 * Signature: (J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_InitializationProcedureSVD_cInitIfaceDispose
(JNIEnv *, jobject, jlong cInitIface)
{
    services::SharedPtr<pca::PartialResultsInitIface<svdDense> > *ptr = (services::SharedPtr<pca::PartialResultsInitIface<svdDense> > *) cInitIface;
    delete ptr;
}
