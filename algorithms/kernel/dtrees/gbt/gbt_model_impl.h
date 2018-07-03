/* file: gbt_model_impl.h */
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

/*
//++
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#ifndef __GBT_MODEL_IMPL__
#define __GBT_MODEL_IMPL__

#include "dtrees_model_impl.h"
#include "algorithms/regression/tree_traverse.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace internal
{

class ModelImpl : public dtrees::internal::ModelImpl
{
public:
    typedef dtrees::internal::ModelImpl ImplType;
    typedef dtrees::internal::TreeImpRegression<> TreeType;

    //Methods common for regression or classification model, not virtual!!!
    size_t numberOfTrees() const;
    void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const;
    void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const;
    void add(dtrees::internal::DecisionTreeTable* pTbl);
    static dtrees::internal::DecisionTreeTable* treeToTable(TreeType& t);
};

} // namespace internal
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
