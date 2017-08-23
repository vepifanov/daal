/* file: ParameterMomentum.java */
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
 * @ingroup sgd
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sgd;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.algorithms.optimization_solver.sgd.BaseParameter;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETERMOMENTUM"></a>
 * @brief ParameterMomentum of the SGD algorithm
 */
public class ParameterMomentum extends BaseParameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * Constructs the parameter for SGD algorithm
    * @param context       Context to manage the SGD algorithm
    */
    public ParameterMomentum(DaalContext context) {
        super(context);
    }

    /**
    * Constructs the parameter for SGD algorithm
    * @param context                Context to manage the SGD algorithm
    * @param cParameterMomentum    Pointer to C++ implementation of the parameter
    */
    public ParameterMomentum(DaalContext context, long cParameterMomentum) {
        super(context, cParameterMomentum);
    }

    /**
    * Sets the number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    * @param batchSize The number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    */
    public void setBatchSize(long batchSize) {
        cSetBatchSize(this.cObject, batchSize);
    }

    /**
    * Returns the number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    * @return The number of batch indices to compute the stochastic gradient.
    * If batchSize is equal to the number of terms in objective
    * function then no random sampling is performed, and all terms are
    * used to calculate the gradient. This parameter is ignored
    * if batchIndices is provided.
    */
    public long getBatchSize() {
        return cGetBatchSize(this.cObject);
    }

    /**
    * Sets the momentum value
    * @param momentum The momentum value
    */
    public void setMomentum(double momentum) {
        cSetMomentum(this.cObject, momentum);
    }

    /**
    * Returns the momentum value
    * @return The momentum value
    */
    public double getMomentum() {
        return cGetMomentum(this.cObject);
    }

    private native void cSetBatchSize(long cObject, long batchSize);
    private native long cGetBatchSize(long cObject);
    private native void cSetMomentum(long cObject, double momentum);
    private native double cGetMomentum(long cObject);
}
/** @} */
