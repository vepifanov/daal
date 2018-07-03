/* file: Model.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @defgroup decision_tree_regression Regression
 * @brief Contains classes for decision tree regression algorithm
 * @ingroup decision_tree
 * @{
 */
/**
 * @brief Contains classes of the decision tree regression algorithm
 */
package com.intel.daal.algorithms.decision_tree.regression;

import com.intel.daal.algorithms.regression.TreeNodeVisitor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__MODEL"></a>
 * @brief %Model trained by decision tree regression algorithm in batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     *  Perform Depth First Traversal of a tree in the model
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseDF(TreeNodeVisitor visitor) {
        cTraverseDF(this.cObject, visitor);
    }

    /**
     *  Perform Breadth First Traversal of a tree in the model
     * @param visitor This object gets notified when tree nodes are visited
     */
    public void traverseBF(TreeNodeVisitor visitor) {
        cTraverseBF(this.cObject, visitor);
    }

    private native void cTraverseDF(long modAddr, TreeNodeVisitor visitorObj);
    private native void cTraverseBF(long modAddr, TreeNodeVisitor visitorObj);
}
/** @} */
