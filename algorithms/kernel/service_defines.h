/* file: service_defines.h */
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
//++
//  Declaration of service constants
//--
*/

#ifndef __SERVICE_DEFINES_H__
#define __SERVICE_DEFINES_H__

#include "services/env_detect.h"

int __daal_serv_cpu_detect(int );

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #define PRAGMA_IVDEP
    #define PRAGMA_NOVECTOR
    #define PRAGMA_VECTOR_ALIGNED
    #define PRAGMA_VECTOR_UNALIGNED
    #define PRAGMA_VECTOR_ALWAYS
    #define PRAGMA_SIMD_ASSERT
    #define PRAGMA_ICC_OMP(ARGS)
    #define PRAGMA_ICC_NO16(ARGS)
#else
    #define PRAGMA_IVDEP _Pragma("ivdep")
    #define PRAGMA_NOVECTOR _Pragma("novector")
    #define PRAGMA_VECTOR_ALIGNED _Pragma("vector aligned")
    #define PRAGMA_VECTOR_UNALIGNED _Pragma("vector unaligned")
    #define PRAGMA_VECTOR_ALWAYS _Pragma("vector always")
    #define PRAGMA_SIMD_ASSERT _Pragma("simd assert")
    #define PRAGMA_ICC_TO_STR(ARGS) _Pragma(#ARGS)
    #define PRAGMA_ICC_OMP(ARGS) PRAGMA_ICC_TO_STR(omp ARGS)
    #define PRAGMA_ICC_NO16(ARGS) PRAGMA_ICC_TO_STR(ARGS)
#endif

#if defined __APPLE__ && defined __INTEL_COMPILER && (__INTEL_COMPILER==1600)
    #undef  PRAGMA_ICC_TO_STR
    #define PRAGMA_ICC_TO_STR(ARGS) _Pragma(#ARGS)
    #undef  PRAGMA_ICC_OMP
    #define PRAGMA_ICC_OMP(ARGS) PRAGMA_ICC_TO_STR(ARGS)
    #undef  PRAGMA_ICC_NO16
    #define PRAGMA_ICC_NO16(ARGS)
#endif

namespace daal
{

/* Execution if(!this->_errors->isEmpty()) */
enum ServiceStatus
{
    SERV_ERR_OK = 0,
    SERV_ERR_MKL_SVD_ITH_PARAM_ILLEGAL_VALUE,
    SERV_ERR_MKL_SVD_XBDSQR_DID_NOT_CONVERGE,
    SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE,
    SERV_ERR_MKL_QR_XBDSQR_DID_NOT_CONVERGE
};

/** Data storage format */
enum DataFormat
{
    dense   = 0,    /*!< Dense storage format */
    CSR     = 1     /*!< Compressed Sparse Rows (CSR) storage format */
};

}

/* CPU comparison macro */
#define __sse2__        (0)
#define __ssse3__       (1)
#define __sse42__       (2)
#define __avx__         (3)
#define __avx2__        (4)
#define __avx512_mic__  (5)
#define __avx512__      (6)

#define __float__        (0)
#define __double__       (1)

#define CPU_sse2        __sse2__
#define CPU_ssse3       __ssse3__
#define CPU_sse42       __sse42__
#define CPU_avx         __avx__
#define CPU_avx2        __avx2__
#define CPU_avx512_mic  __avx512_mic__
#define CPU_avx512      __avx512__

#define FPTYPE_float    __float__
#define FPTYPE_double   __double__

#define __GLUE__(a,b)     a##b
#define __CPUID__(cpu)    __GLUE__(CPU_,cpu)
#define __FPTYPE__(type)  __GLUE__(FPTYPE_,type)

#endif
