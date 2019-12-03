/* file: types.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#ifndef __ONEAPI_INTERNAL_MATH_TYPES_H__
#define __ONEAPI_INTERNAL_MATH_TYPES_H__

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{
namespace interface1
{
enum class Layout
{
    ColMajor,
    RowMajor
};

enum class Transpose
{
    NoTrans,
    Trans
};

enum class UpLo
{
    Upper,
    Lower
};

} // namespace interface1

using interface1::Layout;
using interface1::Transpose;
using interface1::UpLo;

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
