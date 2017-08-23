/* file: LowOrderMomsUserInitialization.java */
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
 //     Initialization procedure for low order moments algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOWORDERMOMSUSERINITIALIZATION">
 * @example LowOrderMomsUserInitialization.java
 */

package com.intel.daal.examples.moments;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;

class LowOrderMomsUserInitialization extends InitializationProcedure {
    /**
     * Constructs the user-defined initialization procedure
     */
    public LowOrderMomsUserInitialization(DaalContext context, int _nFeatures) {
        super(context);
        nFeatures = _nFeatures;
    }

    /**
     * Sets initial parameters of low order moments algorithm
     * @param input         %Input of the algorithm
     * @param partialResult Partial results of the algorithm
     */
    @Override
    public void initialize(Input input, PartialResult partialResult) {
        NumericTable dataTable = input.get(InputId.data);

        DoubleBuffer dataRowDouble = DoubleBuffer.allocate(nFeatures);
        dataRowDouble = dataTable.getBlockOfRows(0, 1, dataRowDouble);

        NumericTable nRowsTable          = partialResult.get(PartialResultId.nObservations);
        NumericTable partialMinimumTable = partialResult.get(PartialResultId.partialMinimum);
        NumericTable partialMaximumTable = partialResult.get(PartialResultId.partialMaximum);
        NumericTable partialSumTable     = partialResult.get(PartialResultId.partialSum);
        NumericTable partialSumSquaresTable         = partialResult.get(PartialResultId.partialSumSquares);
        NumericTable partialSumSquaresCenteredTable = partialResult.get(PartialResultId.partialSumSquaresCentered);

        IntBuffer nRowsBuffer = IntBuffer.allocate(1);
        DoubleBuffer partialMinimumBuffer = DoubleBuffer.allocate(nFeatures);
        DoubleBuffer partialMaximumBuffer = DoubleBuffer.allocate(nFeatures);
        DoubleBuffer partialSumBuffer = DoubleBuffer.allocate(nFeatures);
        DoubleBuffer partialSumSquaresBuffer = DoubleBuffer.allocate(nFeatures);
        DoubleBuffer partialSumSquaresCenteredBuffer = DoubleBuffer.allocate(nFeatures);

        nRowsBuffer          = nRowsTable.getBlockOfRows(0, 1, nRowsBuffer);
        partialMinimumBuffer = partialMinimumTable.getBlockOfRows(0, 1, partialMinimumBuffer);
        partialMaximumBuffer = partialMaximumTable.getBlockOfRows(0, 1, partialMaximumBuffer);
        partialSumBuffer     = partialSumTable.getBlockOfRows(0, 1, partialSumBuffer);
        partialSumSquaresBuffer         = partialSumSquaresTable.getBlockOfRows(0, 1, partialSumSquaresBuffer);
        partialSumSquaresCenteredBuffer = partialSumSquaresCenteredTable.getBlockOfRows(0, 1, partialSumSquaresCenteredBuffer);

        nRowsBuffer.put(0, 0);
        for (int i = 0; i < nFeatures; i++) {
            double value = dataRowDouble.get(i);
            partialMinimumBuffer.put(i, value);
            partialMaximumBuffer.put(i, value);
            partialSumBuffer.put(i, 0.0);
            partialSumSquaresBuffer.put(i, 0.0);
            partialSumSquaresCenteredBuffer.put(i, 0.0);
        }

        nRowsTable.releaseBlockOfRows(0, 1, nRowsBuffer);
        partialMinimumTable.releaseBlockOfRows(0, 1, partialMinimumBuffer);
        partialMaximumTable.releaseBlockOfRows(0, 1, partialMaximumBuffer);
        partialSumTable.releaseBlockOfRows(0, 1, partialSumBuffer);
        partialSumSquaresTable.releaseBlockOfRows(0, 1, partialSumSquaresBuffer);
        partialSumSquaresCenteredTable.releaseBlockOfRows(0, 1, partialSumSquaresCenteredBuffer);

        dataTable.releaseBlockOfRows(0, 1, dataRowDouble);
    }

    protected int nFeatures;
}
