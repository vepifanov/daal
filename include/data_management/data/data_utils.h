/* file: data_utils.h */
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
//  Implementation of data dictionary utilities.
//--
*/

#ifndef __DATA_UTILS_H__
#define __DATA_UTILS_H__

#include <string>
#include <climits>
#include <cfloat>
#include "services/daal_defines.h"

namespace daal
{
namespace data_management
{
/**
 * \brief Contains classes for Intel(R) Data Analytics Acceleration Library numeric types
 */
namespace data_feature_utils
{
/**
 * @ingroup data_model
 * @{
 */
enum IndexNumType
{
    DAAL_FLOAT32 = 0,
    DAAL_FLOAT64 = 1,
    DAAL_INT32_S = 2,
    DAAL_INT32_U = 3,
    DAAL_INT64_S = 4,
    DAAL_INT64_U = 5,
    DAAL_INT8_S  = 6,
    DAAL_INT8_U  = 7,
    DAAL_INT16_S = 8,
    DAAL_INT16_U = 9,
    DAAL_OTHER_T = 10
};
const int NumOfIndexNumTypes = (int)DAAL_OTHER_T;

enum InternalNumType  { DAAL_SINGLE = 0, DAAL_DOUBLE = 1, DAAL_INT32 = 2, DAAL_OTHER = 0xfffffff };
enum PMMLNumType      { DAAL_GEN_FLOAT = 0, DAAL_GEN_DOUBLE = 1, DAAL_GEN_INTEGER = 2, DAAL_GEN_BOOLEAN = 3,
                        DAAL_GEN_STRING = 4, DAAL_GEN_UNKNOWN = 0xfffffff
                      };
enum FeatureType      { DAAL_CATEGORICAL = 0, DAAL_ORDINAL = 1, DAAL_CONTINUOUS = 2 };

/**
 * Convert from a given C++ type to InternalNumType
 * \return Converted numeric type
 */
template<typename T> inline IndexNumType getIndexNumType() { return DAAL_OTHER_T; }
template<> inline IndexNumType getIndexNumType<float>()            { return DAAL_FLOAT32; }
template<> inline IndexNumType getIndexNumType<double>()           { return DAAL_FLOAT64; }
template<> inline IndexNumType getIndexNumType<int>()              { return DAAL_INT32_S; }
template<> inline IndexNumType getIndexNumType<unsigned int>()     { return DAAL_INT32_U; }
template<> inline IndexNumType getIndexNumType<DAAL_INT64>()       { return DAAL_INT64_S; }
template<> inline IndexNumType getIndexNumType<DAAL_UINT64>()      { return DAAL_INT64_U; }
template<> inline IndexNumType getIndexNumType<char>()             { return DAAL_INT8_S;  }
template<> inline IndexNumType getIndexNumType<unsigned char>()    { return DAAL_INT8_U;  }
template<> inline IndexNumType getIndexNumType<short>()            { return DAAL_INT16_S; }
template<> inline IndexNumType getIndexNumType<unsigned short>()   { return DAAL_INT16_U; }

template<> inline IndexNumType getIndexNumType<long>()
{ return (IndexNumType)(DAAL_INT32_S + (sizeof(long) / 4 - 1) * 2); }

#if (defined(__APPLE__) || defined(__MACH__)) && !defined(__x86_64__)
template<> inline IndexNumType getIndexNumType<unsigned long>()
{ return (IndexNumType)(DAAL_INT32_U + (sizeof(unsigned long) / 4 - 1) * 2); }
#endif

#if !(defined(_WIN32) || defined(_WIN64)) && defined(__x86_64__)
template<> inline IndexNumType getIndexNumType<size_t>()
{ return (IndexNumType)(DAAL_INT32_U + (sizeof(size_t) / 4 - 1) * 2); }
#endif

/**
 * \return Internal numeric type
 */
template<typename T>
inline InternalNumType getInternalNumType()          { return DAAL_OTHER;  }
template<>
inline InternalNumType getInternalNumType<int>()     { return DAAL_INT32;  }
template<>
inline InternalNumType getInternalNumType<double>()  { return DAAL_DOUBLE; }
template<>
inline InternalNumType getInternalNumType<float>()   { return DAAL_SINGLE; }

/**
 * \return PMMLNumType
 */
template<typename T>
inline PMMLNumType getPMMLNumType()                { return DAAL_GEN_UNKNOWN; }
template<>
inline PMMLNumType getPMMLNumType<int>()           { return DAAL_GEN_INTEGER; }
template<>
inline PMMLNumType getPMMLNumType<double>()        { return DAAL_GEN_DOUBLE;  }
template<>
inline PMMLNumType getPMMLNumType<float>()         { return DAAL_GEN_FLOAT;   }
template<>
inline PMMLNumType getPMMLNumType<bool>()          { return DAAL_GEN_BOOLEAN; }
template<>
inline PMMLNumType getPMMLNumType<char *>()         { return DAAL_GEN_STRING;  }
template<>
inline PMMLNumType getPMMLNumType<std::string>()   { return DAAL_GEN_STRING;  }

typedef void(*vectorConvertFuncType)(size_t n, void *src, void *dst);
typedef void(*vectorStrideConvertFuncType)(size_t n, void *src, size_t srcByteStride, void *dst, size_t dstByteStride);

DAAL_EXPORT data_feature_utils::vectorConvertFuncType getVectorUpCast(int, int);
DAAL_EXPORT data_feature_utils::vectorConvertFuncType getVectorDownCast(int, int);

DAAL_EXPORT data_feature_utils::vectorStrideConvertFuncType getVectorStrideUpCast(int, int);
DAAL_EXPORT data_feature_utils::vectorStrideConvertFuncType getVectorStrideDownCast(int, int);

/** @} */

} // namespace data_feature_utils
#define DataFeatureUtils data_feature_utils
}
} // namespace daal
#endif
