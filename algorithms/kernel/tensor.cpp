/** file tensor.cpp */
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

#include "services/daal_defines.h"
#include "services/collection.h"
#include "services/env_detect.h"
#include "data_utils.h"
#include "tensor.h"
#include "homogen_tensor.h"
#include "mkl_tensor.h"
#include "service_sort.h"
#include "service_defines.h"
#include "service_dnn.h"
#include "service_dnn_internal.h"

using namespace daal::algorithms::internal;
using namespace daal::internal;

/**
 * Checks the correctness of this tensor
 * \param[in] tensor        Pointer to the tensor to check
 * \param[in] errors        Pointer to the collection of errors
 * \param[in] description   Additional information about error
 * \param[in] dims          Collection with required tensor dimension sizes
 * \return                  Check status:  True if the tensor satisfies the requirements, false otherwise.
 */
bool daal::data_management::checkTensor(const Tensor *tensor, services::ErrorCollection *errors,
                                        const char *description, const services::Collection<size_t> *dims)
{
    using namespace daal::services;

    if (tensor == 0)
    {
        SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorNullTensor));
        error->addStringDetail(ArgumentName, description);
        errors->add(error);
        return false;
    }

    if (tensor->getNumberOfDimensions() == 0)
    {
        SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectNumberOfDimensionsInTensor));
        error->addStringDetail(ArgumentName, description);
        errors->add(error);
        return false;
    }

    if (dims)
    {
        /* Here if collection of the required dimension sizes is provided */
        if (tensor->getNumberOfDimensions() != dims->size())
        {
            SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectNumberOfDimensionsInTensor));
            error->addStringDetail(ArgumentName, description);
            errors->add(error);
            return false;
        }

        for (size_t d = 0; d < dims->size(); d++)
        {
            if (tensor->getDimensionSize(d) != (*dims)[d])
            {
                SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectSizeOfDimensionInTensor));
                error->addStringDetail(ArgumentName, description);
                error->addIntDetail(Dimension, (int)d);
                errors->add(error);
                return false;
            }
        }
    }

    return tensor->check(errors, description);
}

/**
 *  Returns the full size of the tensor in number of elements
 *  \return The full size of the tensor in number of elements
 */
size_t daal::data_management::Tensor::getSize() const
{
        size_t nDim = getNumberOfDimensions();
        if( nDim==0 ) return 0;

        size_t size = 1;

        for(size_t i=0; i<nDim; i++)
        {
            size *= (_layoutPtr->getDimensions())[i];
        }

        return size;
}

/**
 *  Returns the product of sizes of the range of dimensions
 *  \param[in] startingIdx The first dimension to include in the range
 *  \param[in] rangeSize   Number of dimensions to include in the range
 *  \return The product of sizes of the range of dimensions
 */
size_t daal::data_management::Tensor::getSize(size_t startingIdx, size_t rangeSize) const
{
        size_t nDim = getNumberOfDimensions();
        if( nDim==0 || rangeSize==0 || startingIdx>=nDim || startingIdx+rangeSize > nDim ) return 0;

        size_t size = 1;

        for(size_t i=0; i<rangeSize; i++)
        {
            size *= (_layoutPtr->getDimensions())[startingIdx+i];
        }

        return size;
}

namespace daal
{
namespace data_management
{
namespace interface1
{

template<typename DataType> DAAL_EXPORT
SubtensorDescriptor<DataType>::~SubtensorDescriptor()
{
    freeBuffer();
    if( _dimNums != _tensorNDimsBuffer )
    {
        daal::services::daal_free( _dimNums );
    }
    if( _layout && _layoutOwnFlag )
    {
        delete _layout;
    }
}

#define DAAL_INSTANTIATE_SUBTENSORDESCRIPTORDESTRUCTOR(T)          \
template DAAL_EXPORT SubtensorDescriptor<T>::~SubtensorDescriptor();

DAAL_INSTANTIATE_SUBTENSORDESCRIPTORDESTRUCTOR(float )
DAAL_INSTANTIATE_SUBTENSORDESCRIPTORDESTRUCTOR(double)
DAAL_INSTANTIATE_SUBTENSORDESCRIPTORDESTRUCTOR(int   )

void TensorOffsetLayout::shuffleDimensions(const services::Collection<size_t>& dimsOrder)
{
    services::Collection<size_t> newDims   (dimsOrder.size());
    services::Collection<size_t> newOffsets(dimsOrder.size());

    for(size_t i=0;i<_nDims;i++)
    {
        newDims   [ dimsOrder[i] ] = _dims   [i];
        newOffsets[ dimsOrder[i] ] = _offsets[i];
    }
    _dims    = newDims;
    _offsets = newOffsets;
    _indices = dimsOrder;

    checkLayout();
}

void TensorOffsetLayout::checkLayout()
{
    size_t lastIndex = _nDims-1;

    int defaultLayoutMatch = (_offsets[lastIndex] == 1);
    int rawLayoutMatch = (_offsets[lastIndex] == 1);
    for(size_t i=1; i<_nDims; i++)
    {
        defaultLayoutMatch += (_offsets[lastIndex-i] == _offsets[lastIndex-i+1]*_dims[lastIndex-i+1]);
        rawLayoutMatch += (_offsets[lastIndex-i] >= _offsets[lastIndex-i+1]);
    }

    _isDefaultLayout = ( defaultLayoutMatch == _nDims );
    _isRawLayout = ( rawLayoutMatch == _nDims );
}

void TensorOffsetLayout::sortOffsets()
{
    if(_isRawLayout) return;
    indexBubbleSortDesc<size_t, sse2>(_offsets, _dims, _indices);
    checkLayout();
}

template <typename DataType>
template <typename T>
void HomogenTensor<DataType>::getTSubtensor( size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                                             int rwFlag, SubtensorDescriptor<T> &block, const TensorOffsetLayout& layout )
{
    TensorOffsetLayout *pLayout = &_layout;

    if( !pLayout->isLayout(layout) )
    {
        block.saveOffsetLayoutCopy( createDefaultSubtensorLayout() );
        pLayout = const_cast<TensorOffsetLayout*>(block.getLayout());
        pLayout->shuffleDimensions(layout.getIndices());
    }
    else
    {
        block.saveOffsetLayout(_layout);
    }

    size_t        nDim        = pLayout->getDimensions().size();
    const size_t *dimSizes    = &((pLayout->getDimensions())[0]);
    const size_t *_dimOffsets = &((pLayout->getOffsets())[0]);

    size_t blockSize = block.setDetails( nDim, dimSizes, fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwFlag );

    size_t shift = 0;
    for( size_t i = 0; i < fixedDims; i++ )
    {
        shift += fixedDimNums[i] * _dimOffsets[i];
    }
    if( fixedDims != nDim )
    {
        shift += rangeDimIdx * _dimOffsets[fixedDims];
    }

    if( pLayout->isRawLayout() )
    {
        if( IsSameType<T, DataType>::value )
        {
            block.setPtr( (T *)(_ptr + shift) );
        }
        else
        {
            if( !block.resizeBuffer() )
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }

            if( rwFlag & (int)readOnly )
            {
                data_feature_utils::vectorConvertFuncType convertFunc = data_feature_utils::getVectorUpCast(data_feature_utils::getIndexNumType<DataType>(), data_feature_utils::getInternalNumType<T>());
                if (convertFunc)
                    convertFunc( blockSize, _ptr + shift, block.getPtr() );
                else
                    return;

            }
        }
    }
    else
    {
        if( !block.resizeBuffer() )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        if( rwFlag & (int)readOnly )
        {
            size_t leftDims = nDim - fixedDims;

            size_t* bIdxs = new size_t[leftDims];
            size_t* bDims = new size_t[leftDims];

            bIdxs[0] = 0;
            bDims[0] = rangeDimNum;
            for( size_t i=1; i<leftDims; i++ )
            {
                bIdxs[i] = 0;
                bDims[i] = dimSizes[fixedDims+i];
            }

            for( size_t b=0; b<blockSize; b++ )
            {
                size_t rShift = 0;
                for( size_t i=0; i<leftDims; i++ )
                {
                    rShift += bIdxs[ i ]*_dimOffsets[fixedDims+i];
                }

                *(block.getPtr() + b) = *(_ptr + shift + rShift);

                for( size_t i=0; i<leftDims; i++ )
                {
                    bIdxs[ leftDims-1-i ]++;
                    if( bIdxs[ leftDims-1-i ] < bDims[ leftDims-1-i ] ) break;
                    bIdxs[ leftDims-1-i ] = 0;
                }
            }

            delete[] bDims;
            delete[] bIdxs;
        }
    }
}

template <typename DataType>
template <typename T>
void HomogenTensor<DataType>::releaseTSubtensor( SubtensorDescriptor<T> &block )
{
    if( (block.getRWFlag() & (int)writeOnly) && !block.getInplaceFlag() )
    {
        if( block.getLayout()->isDefaultLayout() )
        {
            if( !IsSameType<T, DataType>::value )
            {
                const size_t *_dimOffsets = &((block.getLayout()->getOffsets())[0]);

                size_t nDim = getNumberOfDimensions();

                size_t blockSize = block.getSize();

                size_t fixedDims     = block.getFixedDims();
                size_t *fixedDimNums = block.getFixedDimNums();
                size_t rangeDimIdx   = block.getRangeDimIdx();

                size_t shift = 0;
                for( size_t i = 0; i < fixedDims; i++ )
                {
                    shift += fixedDimNums[i] * _dimOffsets[i];
                }
                if( fixedDims != nDim )
                {
                    shift += rangeDimIdx * _dimOffsets[fixedDims];
                }
                data_feature_utils::vectorConvertFuncType convertFunc = data_feature_utils::getVectorDownCast(data_feature_utils::getIndexNumType<DataType>(), data_feature_utils::getInternalNumType<T>());
                if (convertFunc)
                    convertFunc( blockSize, block.getPtr(), _ptr + shift );
                else
                    return;
            }
        }
        else
        {
            size_t nDim = getNumberOfDimensions();

            const size_t *dimSizes    = &((block.getLayout()->getDimensions())[0]);
            const size_t *_dimOffsets = &((block.getLayout()->getOffsets())[0]);

            size_t blockSize = block.getSize();

            size_t fixedDims     = block.getFixedDims();
            size_t *fixedDimNums = block.getFixedDimNums();
            size_t rangeDimIdx   = block.getRangeDimIdx();
            size_t rangeDimNum   = block.getRangeDimNum();

            size_t shift = 0;
            for( size_t i = 0; i < fixedDims; i++ )
            {
                shift += fixedDimNums[i] * _dimOffsets[i];
            }
            if( fixedDims != nDim )
            {
                shift += rangeDimIdx * _dimOffsets[fixedDims];
            }

            size_t leftDims = nDim - fixedDims;

            size_t* bIdxs = new size_t[leftDims];
            size_t* bDims = new size_t[leftDims];

            bIdxs[0] = 0;
            bDims[0] = rangeDimNum;
            for( size_t i=1; i<leftDims; i++ )
            {
                bIdxs[i] = 0;
                bDims[i] = dimSizes[fixedDims+i];
            }

            for( size_t b=0; b<blockSize; b++ )
            {
                size_t rShift = 0;
                for( size_t i=0; i<leftDims; i++ )
                {
                    rShift += bIdxs[ i ]*_dimOffsets[fixedDims+i];
                }

                *(_ptr + shift + rShift) = *(block.getPtr() + b);

                for( size_t i=0; i<leftDims; i++ )
                {
                    bIdxs[ leftDims-1-i ]++;
                    if( bIdxs[ leftDims-1-i ] < bDims[ leftDims-1-i ] ) break;
                    bIdxs[ leftDims-1-i ] = 0;
                }
            }

            delete[] bDims;
            delete[] bIdxs;
        }
    }
}

#define DAAL_IMPL_GETSUBTENSOR(T1,T2)                                                                                        \
template<>                                                                                                                   \
void HomogenTensor<T1>::getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, \
                  ReadWriteMode rwflag, SubtensorDescriptor<T2> &block, const TensorOffsetLayout& layout )                   \
{                                                                                                                            \
    getTSubtensor<T2>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, block, layout);                             \
}

#define DAAL_IMPL_RELEASESUBTENSOR(T1,T2)                                 \
template<>                                                                \
void HomogenTensor<T1>::releaseSubtensor(SubtensorDescriptor<T2> &block)  \
{                                                                         \
    releaseTSubtensor<T2>(block);                                         \
}

#define DAAL_IMPL_HOMOGENTENSORCONSTRUCTOR(T)                                                                        \
template<>                                                                                                           \
HomogenTensor<T>::HomogenTensor(const services::Collection<size_t> &dims, T *data) : Tensor(&_layout), _layout(dims) \
{                                                                                                                    \
    _ptr = data;                                                                                                     \
    _allocatedSize = 0;                                                                                              \
    if( data )                                                                                                       \
    {                                                                                                                \
        _allocatedSize = getSize();                                                                                  \
        _memStatus = userAllocated;                                                                                  \
    }                                                                                                                \
    size_t nDim = dims.size();                                                                                       \
    if(nDim == 0)                                                                                                    \
    {                                                                                                                \
        this->_errors->add(services::ErrorNullParameterNotSupported);                                                \
        return;                                                                                                      \
    }                                                                                                                \
}

#define DAAL_INSTANTIATE(T1,T2)                                                                                                                         \
template void HomogenTensor<T1>::getTSubtensor( size_t, const size_t *, size_t, size_t, int, SubtensorDescriptor<T2> &, const TensorOffsetLayout& );    \
template void HomogenTensor<T1>::releaseTSubtensor( SubtensorDescriptor<T2> & );                                                                        \
DAAL_IMPL_GETSUBTENSOR(T1,T2)                                                                                                                           \
DAAL_IMPL_RELEASESUBTENSOR(T1,T2)

#define DAAL_INSTANTIATE_THREE(T1)     \
DAAL_INSTANTIATE(T1, double)           \
DAAL_INSTANTIATE(T1, float )           \
DAAL_INSTANTIATE(T1, int   )           \
DAAL_IMPL_HOMOGENTENSORCONSTRUCTOR(T1)

DAAL_INSTANTIATE_THREE(float         )
DAAL_INSTANTIATE_THREE(double        )
DAAL_INSTANTIATE_THREE(int           )
DAAL_INSTANTIATE_THREE(unsigned int  )
DAAL_INSTANTIATE_THREE(DAAL_INT64    )
DAAL_INSTANTIATE_THREE(DAAL_UINT64   )
DAAL_INSTANTIATE_THREE(char          )
DAAL_INSTANTIATE_THREE(unsigned char )
DAAL_INSTANTIATE_THREE(short         )
DAAL_INSTANTIATE_THREE(unsigned short)
DAAL_INSTANTIATE_THREE(unsigned long )

}
}
}

namespace daal
{
namespace internal
{

template <typename DataType>
template <typename T>
void MklTensor<DataType>::getTSubtensor( size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                                             int rwFlag, SubtensorDescriptor<T> &block, const TensorOffsetLayout& layout )
{
    TensorOffsetLayout *pLayout = &_layout;

    if( !pLayout->isLayout(layout) )
    {
        block.saveOffsetLayoutCopy( createDefaultSubtensorLayout() );
        pLayout = const_cast<TensorOffsetLayout*>(block.getLayout());
        pLayout->shuffleDimensions(layout.getIndices());
    }
    else
    {
        block.saveOffsetLayout(_layout);
    }

    size_t        nDim        = pLayout->getDimensions().size();
    const size_t *dimSizes    = &((pLayout->getDimensions())[0]);
    const size_t *_dimOffsets = &((pLayout->getOffsets())[0]);

    size_t blockSize = block.setDetails( nDim, dimSizes, fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwFlag );

    size_t shift = 0;
    for( size_t i = 0; i < fixedDims; i++ )
    {
        shift += fixedDimNums[i] * _dimOffsets[i];
    }
    if( fixedDims != nDim )
    {
        shift += rangeDimIdx * _dimOffsets[fixedDims];
    }

    if( pLayout->isRawLayout() )
    {
        if( IsSameType<T, DataType>::value )
        {
            block.setPtr( (T *)(_ptr + shift) );
        }
        else
        {
            if( !block.resizeBuffer() )
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }

            if( rwFlag & (int)readOnly )
            {
                data_feature_utils::vectorConvertFuncType convertFunc = data_feature_utils::getVectorUpCast(data_feature_utils::getIndexNumType<DataType>(), data_feature_utils::getInternalNumType<T>());
                if (convertFunc)
                    convertFunc( blockSize, _ptr + shift, block.getPtr() );
                else
                    return;
            }
        }
    }
    else
    {
        if( !block.resizeBuffer() )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        if( rwFlag & (int)readOnly )
        {
            size_t leftDims = nDim - fixedDims;

            size_t* bIdxs = new size_t[leftDims];
            size_t* bDims = new size_t[leftDims];

            bIdxs[0] = 0;
            bDims[0] = rangeDimNum;
            for( size_t i=1; i<leftDims; i++ )
            {
                bIdxs[i] = 0;
                bDims[i] = dimSizes[fixedDims+i];
            }

            for( size_t b=0; b<blockSize; b++ )
            {
                size_t rShift = 0;
                for( size_t i=0; i<leftDims; i++ )
                {
                    rShift += bIdxs[ i ]*_dimOffsets[fixedDims+i];
                }

                *(block.getPtr() + b) = *(_ptr + shift + rShift);

                for( size_t i=0; i<leftDims; i++ )
                {
                    bIdxs[ leftDims-1-i ]++;
                    if( bIdxs[ leftDims-1-i ] < bDims[ leftDims-1-i ] ) break;
                    bIdxs[ leftDims-1-i ] = 0;
                }
            }

            delete[] bDims;
            delete[] bIdxs;
        }
    }
}

template <typename DataType>
template <typename T>
void MklTensor<DataType>::releaseTSubtensor( SubtensorDescriptor<T> &block )
{
    if( (block.getRWFlag() & (int)writeOnly) && !block.getInplaceFlag() )
    {
        if( block.getLayout()->isDefaultLayout() )
        {
            if( !IsSameType<T, DataType>::value )
            {
                const size_t *_dimOffsets = &((block.getLayout()->getOffsets())[0]);

                size_t nDim = getNumberOfDimensions();

                size_t blockSize = block.getSize();

                size_t fixedDims     = block.getFixedDims();
                size_t *fixedDimNums = block.getFixedDimNums();
                size_t rangeDimIdx   = block.getRangeDimIdx();

                size_t shift = 0;
                for( size_t i = 0; i < fixedDims; i++ )
                {
                    shift += fixedDimNums[i] * _dimOffsets[i];
                }
                if( fixedDims != nDim )
                {
                    shift += rangeDimIdx * _dimOffsets[fixedDims];
                }
                data_feature_utils::vectorConvertFuncType convertFunc = data_feature_utils::getVectorDownCast(data_feature_utils::getIndexNumType<DataType>(), data_feature_utils::getInternalNumType<T>());
                if (convertFunc)
                    convertFunc( blockSize, block.getPtr(), _ptr + shift );
                else
                    return;
            }
        }
        else
        {
            size_t nDim = getNumberOfDimensions();

            const size_t *dimSizes    = &((block.getLayout()->getDimensions())[0]);
            const size_t *_dimOffsets = &((block.getLayout()->getOffsets())[0]);

            size_t blockSize = block.getSize();

            size_t fixedDims     = block.getFixedDims();
            size_t *fixedDimNums = block.getFixedDimNums();
            size_t rangeDimIdx   = block.getRangeDimIdx();
            size_t rangeDimNum   = block.getRangeDimNum();

            size_t shift = 0;
            for( size_t i = 0; i < fixedDims; i++ )
            {
                shift += fixedDimNums[i] * _dimOffsets[i];
            }
            if( fixedDims != nDim )
            {
                shift += rangeDimIdx * _dimOffsets[fixedDims];
            }

            size_t leftDims = nDim - fixedDims;

            size_t* bIdxs = new size_t[leftDims];
            size_t* bDims = new size_t[leftDims];

            bIdxs[0] = 0;
            bDims[0] = rangeDimNum;
            for( size_t i=1; i<leftDims; i++ )
            {
                bIdxs[i] = 0;
                bDims[i] = dimSizes[fixedDims+i];
            }

            for( size_t b=0; b<blockSize; b++ )
            {
                size_t rShift = 0;
                for( size_t i=0; i<leftDims; i++ )
                {
                    rShift += bIdxs[ i ]*_dimOffsets[fixedDims+i];
                }

                *(_ptr + shift + rShift) = *(block.getPtr() + b);

                for( size_t i=0; i<leftDims; i++ )
                {
                    bIdxs[ leftDims-1-i ]++;
                    if( bIdxs[ leftDims-1-i ] < bDims[ leftDims-1-i ] ) break;
                    bIdxs[ leftDims-1-i ] = 0;
                }
            }

            delete[] bDims;
            delete[] bIdxs;
        }
    }
}

template<typename T>
dnnError_t releaseBufferCpu(T *ptr)
{
    int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
    dnnError_t err;

    switch(cpuid)
    {
        case avx512    : err = Dnn<T, avx512    >::xReleaseBuffer(ptr); break;
        case avx512_mic: err = Dnn<T, avx512_mic>::xReleaseBuffer(ptr); break;
        case avx2      : err = Dnn<T, avx2      >::xReleaseBuffer(ptr); break;
        case avx       : err = Dnn<T, avx       >::xReleaseBuffer(ptr); break;
        case sse42     : err = Dnn<T, sse42     >::xReleaseBuffer(ptr); break;
        case ssse3     : err = Dnn<T, ssse3     >::xReleaseBuffer(ptr); break;
        default        : err = Dnn<T, sse2      >::xReleaseBuffer(ptr); break;
    };

    return err;
}

template<typename T>
dnnError_t allocateBufferCpu(void **ptr, dnnLayout_t dnnLayout)
{
    int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
    dnnError_t err = (dnnError_t)0;

    switch(cpuid)
    {
        case avx512    : err = Dnn<T, avx512    >::xAllocateBuffer(ptr, dnnLayout); break;
        case avx512_mic: err = Dnn<T, avx512_mic>::xAllocateBuffer(ptr, dnnLayout); break;
        case avx2      : err = Dnn<T, avx2      >::xAllocateBuffer(ptr, dnnLayout); break;
        case avx       : err = Dnn<T, avx       >::xAllocateBuffer(ptr, dnnLayout); break;
        case sse42     : err = Dnn<T, sse42     >::xAllocateBuffer(ptr, dnnLayout); break;
        case ssse3     : err = Dnn<T, ssse3     >::xAllocateBuffer(ptr, dnnLayout); break;
        default        : err = Dnn<T, sse2      >::xAllocateBuffer(ptr, dnnLayout); break;
    };

    return err;
}

template<typename T>
bool layoutCompareCpu(dnnLayout_t dnnLayoutSrc, dnnLayout_t dnnLayoutDst)
{
    int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
    bool sameLayout = false;

    switch(cpuid)
    {
        case avx512    : sameLayout = Dnn<T, avx512    >::xLayoutCompare(dnnLayoutSrc, dnnLayoutDst); break;
        case avx512_mic: sameLayout = Dnn<T, avx512_mic>::xLayoutCompare(dnnLayoutSrc, dnnLayoutDst); break;
        case avx2      : sameLayout = Dnn<T, avx2      >::xLayoutCompare(dnnLayoutSrc, dnnLayoutDst); break;
        case avx       : sameLayout = Dnn<T, avx       >::xLayoutCompare(dnnLayoutSrc, dnnLayoutDst); break;
        case sse42     : sameLayout = Dnn<T, sse42     >::xLayoutCompare(dnnLayoutSrc, dnnLayoutDst); break;
        case ssse3     : sameLayout = Dnn<T, ssse3     >::xLayoutCompare(dnnLayoutSrc, dnnLayoutDst); break;
        default        : sameLayout = Dnn<T, sse2      >::xLayoutCompare(dnnLayoutSrc, dnnLayoutDst); break;
    };

    return sameLayout;
}

template<typename T>
dnnError_t layoutDeleteCpu(dnnLayout_t dnnLayout)
{
    int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
    dnnError_t err;

    switch(cpuid)
    {
        case avx512    : err = Dnn<T, avx512    >::xLayoutDelete(dnnLayout); break;
        case avx512_mic: err = Dnn<T, avx512_mic>::xLayoutDelete(dnnLayout); break;
        case avx2      : err = Dnn<T, avx2      >::xLayoutDelete(dnnLayout); break;
        case avx       : err = Dnn<T, avx       >::xLayoutDelete(dnnLayout); break;
        case sse42     : err = Dnn<T, sse42     >::xLayoutDelete(dnnLayout); break;
        case ssse3     : err = Dnn<T, ssse3     >::xLayoutDelete(dnnLayout); break;
        default        : err = Dnn<T, sse2      >::xLayoutDelete(dnnLayout); break;
    };

    return err;
}

template<typename T>
dnnError_t layoutCreateCpu(dnnLayout_t *dnnLayout, size_t nDim, size_t sizes[], size_t strides[])
{
    int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
    dnnError_t err;

    switch(cpuid)
    {
        case avx512    : err = Dnn<T, avx512    >::xLayoutCreate(dnnLayout, nDim, sizes, strides); break;
        case avx512_mic: err = Dnn<T, avx512_mic>::xLayoutCreate(dnnLayout, nDim, sizes, strides); break;
        case avx2      : err = Dnn<T, avx2      >::xLayoutCreate(dnnLayout, nDim, sizes, strides); break;
        case avx       : err = Dnn<T, avx       >::xLayoutCreate(dnnLayout, nDim, sizes, strides); break;
        case sse42     : err = Dnn<T, sse42     >::xLayoutCreate(dnnLayout, nDim, sizes, strides); break;
        case ssse3     : err = Dnn<T, ssse3     >::xLayoutCreate(dnnLayout, nDim, sizes, strides); break;
        default        : err = Dnn<T, sse2      >::xLayoutCreate(dnnLayout, nDim, sizes, strides); break;
    };

    return err;
}

template<typename T>
dnnError_t layoutConvertCpu(T **ptrSrc, dnnLayout_t dnnLayoutSrc, bool bufAllocatedSrc, T **ptrDst, dnnLayout_t dnnLayoutDst, bool bufAllocatedDst)
{
    int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
    dnnError_t err;

    switch(cpuid)
    {
        case avx512    : {
                             LayoutConvertor<T, avx512> cv(ptrSrc, dnnLayoutSrc, bufAllocatedSrc, ptrDst, dnnLayoutDst, bufAllocatedDst);
                             cv.convert();
                             err = cv.err;
                         }
                         break;
        case avx512_mic: {
                             LayoutConvertor<T, avx512_mic> cv(ptrSrc, dnnLayoutSrc, bufAllocatedSrc, ptrDst, dnnLayoutDst, bufAllocatedDst);
                             cv.convert();
                             err = cv.err;
                         }
                         break;
        case avx2      : {
                             LayoutConvertor<T, avx2> cv(ptrSrc, dnnLayoutSrc, bufAllocatedSrc, ptrDst, dnnLayoutDst, bufAllocatedDst);
                             cv.convert();
                             err = cv.err;
                         }
                         break;
        case avx       : {
                             LayoutConvertor<T, avx> cv(ptrSrc, dnnLayoutSrc, bufAllocatedSrc, ptrDst, dnnLayoutDst, bufAllocatedDst);
                             cv.convert();
                             err = cv.err;
                         }
                         break;
        case sse42     : {
                             LayoutConvertor<T, sse42> cv(ptrSrc, dnnLayoutSrc, bufAllocatedSrc, ptrDst, dnnLayoutDst, bufAllocatedDst);
                             cv.convert();
                             err = cv.err;
                         }
                         break;
        case ssse3     : {
                             LayoutConvertor<T, ssse3> cv(ptrSrc, dnnLayoutSrc, bufAllocatedSrc, ptrDst, dnnLayoutDst, bufAllocatedDst);
                             cv.convert();
                             err = cv.err;
                         }
                         break;
        default        : {
                             LayoutConvertor<T, sse2> cv(ptrSrc, dnnLayoutSrc, bufAllocatedSrc, ptrDst, dnnLayoutDst, bufAllocatedDst);
                             cv.convert();
                             err = cv.err;
                         }
                         break;
    };

    return err;
}

#define DAAL_IMPL_MKLTENSOR(T)                                                                                   \
template<>                                                                                                       \
void MklTensor<T>::freeDataMemory()                                                                              \
{                                                                                                                \
    if( getDataMemoryStatus() == internallyAllocated && _ptr != NULL)                                            \
    {                                                                                                            \
        dnnError_t err = releaseBufferCpu<T>(_ptr);                                                              \
        ON_ERR(err);                                                                                             \
    }                                                                                                            \
                                                                                                                 \
    _ptr = NULL;                                                                                                 \
    _memStatus = notAllocated;                                                                                   \
}                                                                                                                \
                                                                                                                 \
template<>                                                                                                       \
void MklTensor<T>::freeDnnLayout() {                                                                             \
    dnnLayout_t dnnLayout = (dnnLayout_t)_dnnLayout;                                                             \
    if (dnnLayout != NULL)                                                                                       \
    {                                                                                                            \
        layoutDeleteCpu<T>(dnnLayout);                                                                           \
    }                                                                                                            \
                                                                                                                 \
    _dnnLayout = NULL;                                                                                           \
}                                                                                                                \
                                                                                                                 \
template<>                                                                                                       \
void MklTensor<T>::setLayout(void* newDnnLayout)                                                                 \
{                                                                                                                \
    dnnLayout_t dnnLayout = (dnnLayout_t)_dnnLayout;                                                             \
    bool sameLayout = false;                                                                                     \
    if (dnnLayout != NULL)                                                                                       \
    {                                                                                                            \
        sameLayout = layoutCompareCpu<T>(dnnLayout, (dnnLayout_t)newDnnLayout);                                  \
    }                                                                                                            \
                                                                                                                 \
    if (!sameLayout && getDataMemoryStatus() != notAllocated) {                                                  \
        T *newPtr = NULL;                                                                                        \
        dnnError_t err = allocateBufferCpu<T>((void**)&newPtr, (dnnLayout_t)newDnnLayout);                       \
        ON_ERR(err);                                                                                             \
        if (dnnLayout != NULL)                                                                                   \
        {                                                                                                        \
            err = layoutConvertCpu<T>(&_ptr, dnnLayout, true, &newPtr, (dnnLayout_t)newDnnLayout, true);         \
            ON_ERR(err);                                                                                         \
        }                                                                                                        \
                                                                                                                 \
        freeDataMemory();                                                                                        \
                                                                                                                 \
        _ptr = newPtr;                                                                                           \
        _memStatus = internallyAllocated;                                                                        \
    }                                                                                                            \
    freeDnnLayout();                                                                                             \
    _dnnLayout = newDnnLayout;                                                                                   \
    _isPlainLayout = false;                                                                                      \
}                                                                                                                \
                                                                                                                 \
template<>                                                                                                       \
void MklTensor<T>::allocateDataMemory(daal::MemType type)                                                        \
{                                                                                                                \
    freeDataMemory();                                                                                            \
                                                                                                                 \
    if( _memStatus != notAllocated )                                                                             \
    {                                                                                                            \
        /* Error is already reported by freeDataMemory() */                                                      \
        return;                                                                                                  \
    }                                                                                                            \
                                                                                                                 \
    dnnError_t err = allocateBufferCpu<T>((void**)&_ptr, (dnnLayout_t)_dnnLayout);                               \
    ON_ERR(err);                                                                                                 \
                                                                                                                 \
    if( _ptr == 0 )                                                                                              \
    {                                                                                                            \
        this->_errors->add(services::ErrorMemoryAllocationFailed);                                               \
        return;                                                                                                  \
    }                                                                                                            \
                                                                                                                 \
    _memStatus = internallyAllocated;                                                                            \
}                                                                                                                \
                                                                                                                 \
template<>                                                                                                       \
void MklTensor<T>::setPlainLayout()                                                                              \
{                                                                                                                \
    if (_isPlainLayout) {                                                                                        \
        return;                                                                                                  \
    }                                                                                                            \
    const services::Collection<size_t> & dims    = _layout.getDimensions();                                      \
    const services::Collection<size_t> & offsets = _layout.getOffsets();                                         \
                                                                                                                 \
    size_t nDim = dims.size();                                                                                   \
                                                                                                                 \
    size_t *newSizes = new size_t[nDim];                                                                         \
    size_t *newStrides = new size_t[nDim];                                                                       \
                                                                                                                 \
    for(size_t i = 0; i < nDim; i++)                                                                             \
    {                                                                                                            \
        newSizes[i] = dims[nDim - i - 1];                                                                        \
        newStrides[i] = offsets[nDim - i - 1];                                                                   \
    }                                                                                                            \
                                                                                                                 \
    dnnLayout_t newDnnLayout;                                                                                    \
    dnnError_t err = layoutCreateCpu<T>(&newDnnLayout, nDim, newSizes, newStrides);                              \
    ON_ERR(err);                                                                                                 \
    delete [] newSizes;                                                                                          \
    delete [] newStrides;                                                                                        \
                                                                                                                 \
    setLayout((void*)newDnnLayout);                                                                              \
                                                                                                                 \
    _isPlainLayout = true;                                                                                       \
}                                                                                                                \
                                                                                                                 \
template<>                                                                                                       \
MklTensor<T>::MklTensor(size_t nDim, const size_t *dimSizes) :                                                   \
    Tensor(&_layout), _layout(services::Collection<size_t>(nDim, dimSizes)), _ptr(0),                            \
    _dnnLayout(NULL), _isPlainLayout(false)                                                                      \
{                                                                                                                \
    if(!dimSizes)                                                                                                \
    {                                                                                                            \
        this->_errors->add(services::ErrorNullParameterNotSupported);                                            \
        return;                                                                                                  \
    }                                                                                                            \
                                                                                                                 \
    setPlainLayout();                                                                                            \
                                                                                                                 \
    allocateDataMemory();                                                                                        \
}                                                                                                                \
                                                                                                                 \
template<>                                                                                                       \
MklTensor<T>::MklTensor(size_t nDim, const size_t *dimSizes, AllocationFlag memoryAllocationFlag) :              \
    Tensor(&_layout), _layout(services::Collection<size_t>(nDim, dimSizes)), _ptr(0),                            \
    _dnnLayout(NULL), _isPlainLayout(false)                                                                      \
{                                                                                                                \
    if(!dimSizes)                                                                                                \
    {                                                                                                            \
        this->_errors->add(services::ErrorNullParameterNotSupported);                                            \
        return;                                                                                                  \
    }                                                                                                            \
                                                                                                                 \
    setPlainLayout();                                                                                            \
                                                                                                                 \
    if (memoryAllocationFlag == doAllocate)                                                                      \
    {                                                                                                            \
        allocateDataMemory();                                                                                    \
    }                                                                                                            \
}                                                                                                                \
                                                                                                                 \
template<>                                                                                                       \
MklTensor<T>::MklTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag) :         \
    Tensor(&_layout), _ptr(0), _layout(dims), _dnnLayout(NULL), _isPlainLayout(false)                            \
{                                                                                                                \
    setPlainLayout();                                                                                            \
                                                                                                                 \
    if( memoryAllocationFlag == doAllocate )                                                                     \
    {                                                                                                            \
        allocateDataMemory();                                                                                    \
    }                                                                                                            \
}                                                                                                                \
                                                                                                                 \
template<>                                                                                                       \
MklTensor<T>::MklTensor(const services::Collection<size_t> &dims) :                                              \
    Tensor(&_layout), _ptr(0), _layout(dims), _dnnLayout(NULL), _isPlainLayout(false)                            \
{                                                                                                                \
    setPlainLayout();                                                                                            \
                                                                                                                 \
    allocateDataMemory();                                                                                        \
}

#define DAAL_IMPL_MKL_GETSUBTENSOR(T1,T2)                                                                                    \
template<>                                                                                                                   \
void MklTensor<T1>::getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,     \
                  ReadWriteMode rwflag, SubtensorDescriptor<T2> &block, const TensorOffsetLayout& layout )                   \
{                                                                                                                            \
    setPlainLayout();                                                                                                        \
                                                                                                                             \
    getTSubtensor<T2>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, block, layout);                             \
}

#define DAAL_IMPL_MKL_RELEASESUBTENSOR(T1,T2)                             \
template<>                                                                \
void MklTensor<T1>::releaseSubtensor(SubtensorDescriptor<T2> &block)      \
{                                                                         \
    releaseTSubtensor<T2>(block);                                         \
}

#define DAAL_MKL_INSTANTIATE(T1,T2)                                                                                                                 \
template void MklTensor<T1>::getTSubtensor( size_t, const size_t *, size_t, size_t, int, SubtensorDescriptor<T2> &, const TensorOffsetLayout& );    \
template void MklTensor<T1>::releaseTSubtensor( SubtensorDescriptor<T2> & );                                                                        \
DAAL_IMPL_MKL_GETSUBTENSOR(T1,T2)                                                                                                                   \
DAAL_IMPL_MKL_RELEASESUBTENSOR(T1,T2)

#define DAAL_MKL_INSTANTIATE_THREE(T1) \
DAAL_IMPL_MKLTENSOR(T1)                \
DAAL_MKL_INSTANTIATE(T1, double)       \
DAAL_MKL_INSTANTIATE(T1, float )       \
DAAL_MKL_INSTANTIATE(T1, int   )

DAAL_MKL_INSTANTIATE_THREE(float         )
DAAL_MKL_INSTANTIATE_THREE(double        )

}
}
