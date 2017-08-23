/* file: CovUserInitialization.java */
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
 //     Initialization procedure for variance-covariance matrix computation algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-COVUSERINITIALIZATION">
 * @example CovUserInitialization.java
 */

package com.intel.daal.examples.covariance;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;

class CovUserInitialization extends InitializationProcedure {
    /**
     * Constructs the user-defined initialization procedure
     */
    public CovUserInitialization(DaalContext context, int _nFeatures) {
        super(context);
        nFeatures = _nFeatures;
    }

    /**
     * Sets initial parameters of uvariance-covariance matrix computation algorithm
     * @param input         %Input of the algorithm
     * @param partialResult Partial results of the algorithm
     */
    @Override
    public void initialize(Input input, PartialResult partialResult) {

        NumericTable nObservationsTable = partialResult.get(PartialResultId.nObservations);
        NumericTable crossProductTable = partialResult.get(PartialResultId.crossProduct);
        NumericTable sumTable = partialResult.get(PartialResultId.sum);

        IntBuffer nObservationsBuffer = IntBuffer.allocate(1);
        DoubleBuffer crossProductBuffer = DoubleBuffer.allocate(nFeatures * nFeatures);
        DoubleBuffer sumBuffer = DoubleBuffer.allocate(nFeatures);

        nObservationsBuffer = nObservationsTable.getBlockOfRows(0, 1, nObservationsBuffer);
        crossProductBuffer = crossProductTable.getBlockOfRows(0, (long)nFeatures, crossProductBuffer);
        sumBuffer = sumTable.getBlockOfRows(0, 1, sumBuffer);

        nObservationsBuffer.put(0, 0);
        for (int i = 0; i < nFeatures * nFeatures; i++) {
            crossProductBuffer.put(i, 0.0);
        }
        for (int i = 0; i < nFeatures; i++) {
            sumBuffer.put(i, 0.0);
        }

        nObservationsTable.releaseBlockOfRows(0, 1, nObservationsBuffer);
        crossProductTable.releaseBlockOfRows(0, (long)nFeatures, crossProductBuffer);
        sumTable.releaseBlockOfRows(0, 1, sumBuffer);
    }

    protected int nFeatures;
}
