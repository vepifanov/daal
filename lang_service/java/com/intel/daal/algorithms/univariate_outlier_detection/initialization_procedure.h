/* file: initialization_procedure.h */
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
//  Implementation of the class that connects outlier detection Java initialization procedure
//  to C++ algorithm
//--
*/
#ifndef __INITIALIZATION_PROCEDURE_H__
#define __INITIALIZATION_PROCEDURE_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "algorithms/outlier_detection/outlier_detection_univariate_types.h"
#include "java_callback.h"

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{

using namespace daal::data_management;
using namespace daal::services;

struct JavaInit : public InitIface, public JavaCallback
{
    JavaInit(JavaVM *_jvm, jobject _javaObject) : JavaCallback(_jvm, _javaObject) {}

    virtual ~JavaInit()
    {}

    virtual void operator()(NumericTable *data, NumericTable *location, NumericTable *scatter, NumericTable *threshold)
    {
        ThreadLocalStorage tls = _tls.local();
        jint status = jvm->AttachCurrentThread((void **)(&tls.jniEnv), NULL);
        JNIEnv *env = tls.jniEnv;

        /* Get current context */
        jclass javaObjectClass = env->GetObjectClass(javaObject);
        if (javaObjectClass == NULL)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Couldn't find class of this java object");
        }

        jmethodID getContextMethodID = env->GetMethodID(javaObjectClass, "getContext", "()Lcom/intel/daal/services/DaalContext;");
        if (getContextMethodID == NULL)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Couldn't find getContext method");
        }
        jobject context = env->CallObjectMethod(javaObject, getContextMethodID);

        SerializationIfacePtr *dataPtr      = new SerializationIfacePtr(data, EmptyDeleter<SerializationIface>());
        SerializationIfacePtr *locationPtr  = new SerializationIfacePtr(location, EmptyDeleter<SerializationIface>());
        SerializationIfacePtr *scatterPtr   = new SerializationIfacePtr(scatter, EmptyDeleter<SerializationIface>());
        SerializationIfacePtr *thresholdPtr = new SerializationIfacePtr(threshold, EmptyDeleter<SerializationIface>());
        /* Get java numeric table class objects */
        jobject dataTable      = constructJavaObjectFromFactory(env, (jlong)(dataPtr), context);
        jobject locationTable  = constructJavaObjectFromFactory(env, (jlong)(locationPtr), context);
        jobject scatterTable   = constructJavaObjectFromFactory(env, (jlong)(scatterPtr), context);
        jobject thresholdTable = constructJavaObjectFromFactory(env, (jlong)(thresholdPtr), context);

        /* Get ID of inintialize method of initialization procedure class */
        jmethodID initializeMethodID = env->GetMethodID(javaObjectClass, "initialize",
                  "(Lcom/intel/daal/data_management/data/NumericTable;Lcom/intel/daal/data_management/data/NumericTable;Lcom/intel/daal/data_management/data/NumericTable;Lcom/intel/daal/data_management/data/NumericTable;)V");
        if (initializeMethodID == NULL)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Couldn't find Initialize method");
        }
        /* Call inintialize method of initialization procedure class */
        env->CallVoidMethod(javaObject, initializeMethodID, dataTable, locationTable, scatterTable, thresholdTable);

        if(!tls.is_main_thread)
        {
            status = jvm->DetachCurrentThread();
        }
        _tls.local() = tls;
    }
};

} // namespace daal::algorithms::univariate_outlier_detection
} // namespace daal::algorithms
} // namespace daal

#endif
