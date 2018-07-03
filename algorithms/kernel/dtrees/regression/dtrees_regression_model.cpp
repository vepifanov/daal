/* file: dtrees_regression_model.cpp */
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
//  Implementation of the regression model methods common for dtrees
//--
*/

#include "dtrees_model_impl_common.h"
#include "algorithms/regression/tree_traverse.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{

typedef daal::algorithms::dtrees::internal::TreeNodeRegression<RegressionFPType>::Leaf TLeaf;
typedef daal::algorithms::regression::TreeNodeVisitor TVisitor;

namespace dtrees
{
namespace internal
{
template<>
void writeLeaf(const TLeaf& l, DecisionTreeNode& row)
{
    row.featureValueOrResponse = l.response;
}

template<>
bool visitLeaf(size_t level, const DecisionTreeNode& row, TVisitor& visitor)
{
    return visitor.onLeafNode(level, row.featureValueOrResponse);
}

} // namespace dtrees
} // namespace internal
} // namespace algorithms
} // namespace daal
