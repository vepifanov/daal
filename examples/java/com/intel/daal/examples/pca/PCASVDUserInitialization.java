/* file: PCASVDUserInitialization.java */
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
 * <a name="DAAL-EXAMPLE-JAVA-PCASVDUSERINITIALIZATION">
 * @example PCASVDUserInitialization.java
 */

package com.intel.daal.examples.pca;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import com.intel.daal.algorithms.pca.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;

class PCASVDUserInitialization extends InitializationProcedureSVD {
    /**
     * Constructs the user-defined initialization procedure
     */
    public PCASVDUserInitialization(DaalContext context, int _nFeatures) {
        super(context);
        nFeatures = _nFeatures;
    }

    /**
     * Sets initial parameters of uvariance-covariance matrix computation algorithm
     * @param input         Input objects for the PCA algorithm
     * @param partialResult Partial results of the PCA algorithm
     */
    @Override
    public void initialize(Input input, com.intel.daal.algorithms.PartialResult partialResult) {
        PartialSVDResult partialSVDResult = (PartialSVDResult) partialResult;

        NumericTable nObservationsSVDTable = partialSVDResult.get(PartialSVDTableResultID.nObservations);
        NumericTable sumSVDTable           = partialSVDResult.get(PartialSVDTableResultID.sumSVD);
        NumericTable sumSquaresSVDTable    = partialSVDResult.get(PartialSVDTableResultID.sumSquaresSVD);

        IntBuffer nObservationsSVDBuffer = IntBuffer.allocate(1);
        DoubleBuffer sumSVDBuffer        = DoubleBuffer.allocate(nFeatures);
        DoubleBuffer sumSquaresSVDBuffer = DoubleBuffer.allocate(nFeatures);

        nObservationsSVDBuffer = nObservationsSVDTable.getBlockOfRows(0, 1, nObservationsSVDBuffer);
        sumSVDBuffer           = sumSVDTable.getBlockOfRows(0, 1, sumSVDBuffer);
        sumSquaresSVDBuffer    = sumSquaresSVDTable.getBlockOfRows(0, 1, sumSquaresSVDBuffer);

        nObservationsSVDBuffer.put(0, 0);
        for (int i = 0; i < nFeatures; i++) {
            sumSVDBuffer.put(i, 0.0);
            sumSquaresSVDBuffer.put(i, 0.0);
        }

        nObservationsSVDTable.releaseBlockOfRows(0, 1, nObservationsSVDBuffer);
        sumSVDTable.releaseBlockOfRows(0, 1, sumSVDBuffer);
        sumSquaresSVDTable.releaseBlockOfRows(0, 1, sumSquaresSVDBuffer);
    }

    protected int nFeatures;
}
