/* file: decision_tree_regression_model_impl.h */
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
//  Implementation of the class defining the Decision tree model
//--
*/

#ifndef __DECISION_TREE_REGRESSION_MODEL_IMPL_
#define __DECISION_TREE_REGRESSION_MODEL_IMPL_

#include "algorithms/decision_tree/decision_tree_regression_model.h"
#include "regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace interface1
{

struct DecisionTreeNode
{
    size_t dimension;
    size_t leftIndex;
    double cutPointOrDependantVariable;
};

class DecisionTreeTable : public data_management::AOSNumericTable
{
public:
    DecisionTreeTable(size_t rowCount, services::Status &st) : data_management::AOSNumericTable(sizeof(DecisionTreeNode), 3, rowCount, st)
    {
        setFeature<size_t> (0, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, dimension));
        setFeature<size_t> (1, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, leftIndex));
        setFeature<double> (2, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, cutPointOrDependantVariable));
        st |= allocateDataMemory();
    }
    DecisionTreeTable(services::Status &st) : DecisionTreeTable(0, st) {}
};

typedef services::SharedPtr<DecisionTreeTable> DecisionTreeTablePtr;
typedef services::SharedPtr<const DecisionTreeTable> DecisionTreeTableConstPtr;

class Model::ModelImpl : public algorithms::regression::internal::ModelImpl
{
    typedef services::Collection<size_t> NodeIdxArray;
public:
    /**
     * Constructs decision tree model with the specified number of features
     * \param[in] featureCount Number of features
     */
    ModelImpl() : _TreeTable() {}

    /**
     * Returns the decision tree table
     * \return decision tree table
     */
    DecisionTreeTablePtr getTreeTable() { return _TreeTable; }

    /**
     * Returns the decision tree table
     * \return decision tree table
     */
    DecisionTreeTableConstPtr getTreeTable() const { return _TreeTable; }

    /**
     *  Sets a decision tree table
     *  \param[in]  value  decision tree table
     */
    void setTreeTable(const DecisionTreeTablePtr & value) { _TreeTable = value; }

    void traverseDF(algorithms::regression::TreeNodeVisitor& visitor) const
    {
        const DecisionTreeNode* aNode = (const DecisionTreeNode*)_TreeTable->getArray();
        if(aNode)
            traverseNodesDF<algorithms::regression::TreeNodeVisitor>(0, 0, aNode, visitor);
    }

    void traverseBF(algorithms::regression::TreeNodeVisitor& visitor) const
    {
        const DecisionTreeNode* aNode = (const DecisionTreeNode*)_TreeTable->getArray();
        NodeIdxArray aCur;//nodes of current layer
        NodeIdxArray aNext;//nodes of next layer
        if(aNode)
        {
            aCur.push_back(0);
            traverseNodesBF<algorithms::regression::TreeNodeVisitor>(0, aCur, aNext, aNode, visitor);
        }
    }

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        algorithms::regression::internal::ModelImpl::serialImpl<Archive, onDeserialize>(arch);
        arch->setSharedPtrObj(_TreeTable);

        return services::Status();
    }

private:
    DecisionTreeTablePtr _TreeTable;

    template <typename TVisitor>
    bool traverseNodesDF(size_t level, size_t iRowInTable, const DecisionTreeNode* aNode,
        TVisitor& visitor) const
    {
        const DecisionTreeNode& n = aNode[iRowInTable];
        if(n.dimension != static_cast<size_t>(-1))
        {
            if(!visitor.onSplitNode(level, n.dimension, n.cutPointOrDependantVariable))
                return false; //do not continue traversing
            ++level;
            size_t leftIdx = n.leftIndex; size_t rightIdx = leftIdx + 1;
            if(!traverseNodesDF(level, leftIdx, aNode, visitor))
                return false;
            return traverseNodesDF(level, rightIdx, aNode, visitor);
        }
        return visitor.onLeafNode(level, n.cutPointOrDependantVariable);
    }

    template <typename TVisitor>
    bool traverseNodesBF(size_t level, NodeIdxArray& aCur,
        NodeIdxArray& aNext, const DecisionTreeNode* aNode, TVisitor& visitor) const
    {
        for(size_t i = 0; i < aCur.size(); ++i)
        {
            for(size_t j = 0; j < (level ? 2 : 1); ++j)
            {
                const DecisionTreeNode& n = aNode[aCur[i] + j];
                if(n.dimension != static_cast<size_t>(-1))
                {
                    if(!visitor.onSplitNode(level, n.dimension, n.cutPointOrDependantVariable))
                        return false; //do not continue traversing
                    aNext.push_back(n.leftIndex);
                }
                else
                {
                    if(!visitor.onLeafNode(level, n.cutPointOrDependantVariable))
                        return false; //do not continue traversing
                }
            }
        }
        aCur.clear();
        if(!aNext.size())
            return true;//done
        return traverseNodesBF(level + 1, aNext, aCur, aNode, visitor);
    }
};

} // namespace interface1

using interface1::DecisionTreeTable;
using interface1::DecisionTreeNode;
using interface1::DecisionTreeTablePtr;
using interface1::DecisionTreeTableConstPtr;

} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
