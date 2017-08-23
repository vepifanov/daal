/* file: InitializationProcedureIface.java */
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

/**
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INITIALIZATIONPROCEDUREIFACE"></a>
 * @brief Abstract interface class for initialization of partial results for the correlation or variance-covariance matrix algorithm
 */
public abstract class InitializationProcedureIface extends SerializableBase {

    /**
     * Constructs the initialization procedure iface
     * @param context   Context to manage the algorithm
     */
    public InitializationProcedureIface(DaalContext context) {
        super(context);
    }

    /**
     * Initializes partial results of the correlation or variance-covariance matrix algorithm
     * @param input         %Input of the algorithm
     * @param partialResult Partial results of the algorithm
     */
    abstract public void initialize(Input input, PartialResult partialResult);
}
/** @} */
