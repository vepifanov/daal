/* file: mkl_tensor.h */
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
//  Declaration and implementation of the base class for numeric n-cubes.
//--
*/


#ifndef __MKL_TENSOR_H__
#define __MKL_TENSOR_H__

#include "services/daal_defines.h"
#include "data_management/data/tensor.h"

using namespace daal::data_management;

namespace daal
{
namespace internal
{

template<typename DataType = double>
class DAAL_EXPORT MklTensor : public Tensor
{
public:
    DAAL_CAST_OPERATOR(MklTensor<DataType>)

    /** \private */
    MklTensor() : Tensor(&_layout), _layout(services::Collection<size_t>()), _ptr(0), _dnnLayout(NULL), _isPlainLayout(false)
    {
    }

    MklTensor(size_t nDim, const size_t *dimSizes);

    MklTensor(size_t nDim, const size_t *dimSizes, AllocationFlag memoryAllocationFlag);

    MklTensor(const services::Collection<size_t> &dims);

    MklTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag);

    /** \private */
    virtual ~MklTensor()
    {
        freeDataMemory();
        freeDnnLayout();
    }

    DataType* getArray()
    {
        return _ptr;
    }

    void* getLayout()
    {
        return _dnnLayout;
    }

    void setLayout(void* dnnLayout);

    void setPlainLayout();

    bool isPlainLayout()
    {
        return _isPlainLayout;
    }

    TensorOffsetLayout& getTensorLayout()
    {
        return _layout;
    }

    virtual TensorOffsetLayout createDefaultSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        return TensorOffsetLayout(_layout);
    }

    virtual TensorOffsetLayout createRawSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        TensorOffsetLayout layout(_layout);
        layout.sortOffsets();
        return layout;
    }

    virtual void setDimensions(size_t nDim, const size_t *dimSizes) DAAL_C11_OVERRIDE
    {
        if(!dimSizes)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        _layout = TensorOffsetLayout(services::Collection<size_t>(nDim, dimSizes));
        setPlainLayout();
    }

    virtual void setDimensions(const services::Collection<size_t>& dimensions) DAAL_C11_OVERRIDE
    {
        if(!dimensions.size())
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        _layout = TensorOffsetLayout(dimensions);
        setPlainLayout();
    }

    virtual void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE;

    void assign(const DataType initValue)
    {
        setPlainLayout();

        size_t size = getSize();

        for(size_t i = 0; i < size; i++)
        {
            _ptr[i] = initValue;
        }
    }

    virtual void freeDataMemory() DAAL_C11_OVERRIDE;

    void freeDnnLayout();

    void getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<double> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;
    void getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<float> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;
    void getSubtensorEx(size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                        ReadWriteMode rwflag, SubtensorDescriptor<int> &block,
                        const TensorOffsetLayout& layout) DAAL_C11_OVERRIDE;

    void getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<double>& subtensor ) DAAL_C11_OVERRIDE
    {
        getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    void getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<float>& subtensor ) DAAL_C11_OVERRIDE
    {
        getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    void getSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<int>& subtensor ) DAAL_C11_OVERRIDE
    {
        getSubtensorEx(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, _layout );
    }

    void releaseSubtensor(SubtensorDescriptor<double> &block) DAAL_C11_OVERRIDE;
    void releaseSubtensor(SubtensorDescriptor<float>  &block) DAAL_C11_OVERRIDE;
    void releaseSubtensor(SubtensorDescriptor<int>    &block) DAAL_C11_OVERRIDE;

    virtual services::SharedPtr<Tensor> getSampleTensor(size_t firstDimIndex) DAAL_C11_OVERRIDE
    {
        setPlainLayout();

        services::Collection<size_t> newDims = getDimensions();
        if(!_ptr || newDims.size() == 0 || newDims[0] <= firstDimIndex) { return services::SharedPtr<Tensor>(); }
        newDims[0] = 1;
        const size_t *_dimOffsets = &((_layout.getOffsets())[0]);
        return services::SharedPtr<Tensor>(new HomogenTensor<DataType>(newDims, _ptr + _dimOffsets[0]*firstDimIndex));
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return daal::data_management::data_feature_utils::getIndexNumType<DataType>() + SERIALIZATION_MKL_TENSOR_ID;
    }

protected:
    void serializeImpl  (InputDataArchive  *archive) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( archive );}

    void deserializeImpl(OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( archive );}

    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *archive )
    {
        Tensor::serialImpl<Archive, onDeserialize>( archive );

        archive->setObj( &_layout );

        setPlainLayout();

        bool isAllocated = (_memStatus != notAllocated);
        archive->set( isAllocated );

        if( onDeserialize )
        {
            freeDataMemory();

            if( isAllocated )
            {
                allocateDataMemory();
            }
        }

        if(_memStatus != notAllocated)
        {
            archive->set( _ptr, getSize() );
        }
    }

private:
    template <typename T>
    void getTSubtensor( size_t fixedDims, const size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, int rwFlag,
                        SubtensorDescriptor<T> &block, const TensorOffsetLayout& layout );
    template <typename T>
    void releaseTSubtensor( SubtensorDescriptor<T> &block );

private:
    DataType           *_ptr;
    TensorOffsetLayout  _layout;
    void               *_dnnLayout;
    bool                _isPlainLayout;
};

}
} // namespace daal

#endif
