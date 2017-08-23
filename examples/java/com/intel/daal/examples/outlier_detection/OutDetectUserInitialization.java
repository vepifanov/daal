/* file: OutDetectUserInitialization.java */
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
 //  Content:
 //     Initialization procedure for univariate outlier detection algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-OUTDETECTUSERINITIALIZATION">
 * @example OutDetectUserInitialization.java
 */

package com.intel.daal.examples.outlier_detection;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import com.intel.daal.algorithms.univariate_outlier_detection.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

class OutDetectUserInitialization extends InitializationProcedure {
    /**
     * Constructs the user-defined initialization procedure
     */
    public OutDetectUserInitialization(DaalContext context, int _nFeatures) {
        super(context);
        nFeatures = _nFeatures;
    }

    /**
     * Sets initial parameters of univariate outlier detection algorithm
     * @param data        %Input data table of size n x p
     * @param location    Vector of mean estimates of size 1 x p
     * @param scatter     Measure of spread, the variance-covariance matrix of size 1 x p
     * @param threshold   Limit that defines the outlier region, the array of size 1 x p containing a non-negative number
     */
    @Override
    public void initialize(NumericTable data, NumericTable location, NumericTable scatter, NumericTable threshold) {
        DoubleBuffer locationBuffer  = DoubleBuffer.allocate(nFeatures);
        DoubleBuffer scatterBuffer   = DoubleBuffer.allocate(nFeatures);
        DoubleBuffer thresholdBuffer = DoubleBuffer.allocate(nFeatures);

        locationBuffer  = location.getBlockOfRows(0, 1, locationBuffer);
        scatterBuffer   = scatter.getBlockOfRows(0, 1, scatterBuffer);
        thresholdBuffer = threshold.getBlockOfRows(0, 1, thresholdBuffer);

        for (int i = 0; i < nFeatures; i++) {
            locationBuffer.put(i, 0.0);
            scatterBuffer.put(i, 1.0);
            thresholdBuffer.put(i, 3.0);
        }

        location.releaseBlockOfRows(0, 1, locationBuffer);
        scatter.releaseBlockOfRows(0, 1, scatterBuffer);
        threshold.releaseBlockOfRows(0, 1, thresholdBuffer);
    }

    protected int nFeatures;
}
