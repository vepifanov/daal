/* file: gbt_model.cpp */
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

#include "gbt_model_impl.h"
#include "dtrees_model_impl_common.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace internal
{
typedef gbt::internal::ModelImpl::TreeType::NodeType::Leaf TLeaf;
typedef daal::algorithms::regression::TreeNodeVisitor TVisitor;

size_t ModelImpl::numberOfTrees() const
{
    return ImplType::size();
}

void ModelImpl::traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const
{
    if(iTree >= size())
        return;
    const DecisionTreeTable& t = *at(iTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t.getArray();
    if(aNode)
        traverseNodeDF<TVisitor>(0, 0, aNode, visitor);
}

void ModelImpl::traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const
{
    if(iTree >= size())
        return;
    const DecisionTreeTable& t = *at(iTree);
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t.getArray();
    NodeIdxArray aCur;//nodes of current layer
    NodeIdxArray aNext;//nodes of next layer
    if(aNode)
    {
        aCur.push_back(0);
        traverseNodesBF<TVisitor>(0, aCur, aNext, aNode, visitor);
    }
}

dtrees::internal::DecisionTreeTable* ModelImpl::treeToTable(TreeType& t)
{
    return t.convertToTable();
}

void ModelImpl::add(dtrees::internal::DecisionTreeTable* pTbl)
{
    DAAL_ASSERT(pTbl);
    _nTree.inc();
    _serializationData->push_back(SerializationIfacePtr(pTbl));
}

} // namespace internal
} // namespace gbt
} // namespace algorithms
} // namespace daal
