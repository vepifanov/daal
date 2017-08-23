/* file: service_numeric_table.h */
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
//  CPU-specified homogeneous numeric table
//--
*/

#ifndef __SERVICE_NUMERIC_TABLE_H__
#define __SERVICE_NUMERIC_TABLE_H__

#include "homogen_numeric_table.h"
#include "csr_numeric_table.h"
#include "service_defines.h"
#include "service_memory.h"

using namespace daal::data_management;

namespace daal
{
namespace internal
{

template <CpuType cpu>
class NumericTableFeatureCPU : public NumericTableFeature
{
public:
    NumericTableFeatureCPU() : NumericTableFeature() {}
    virtual ~NumericTableFeatureCPU() {}
};

template <CpuType cpu>
class NumericTableDictionaryCPU : public NumericTableDictionary
{
public:
    NumericTableDictionaryCPU( size_t nfeat )
    {
        _nfeat = 0;
        _dict  = (NumericTableFeature *)(new NumericTableFeatureCPU<cpu>[1]);
        _featuresEqual = DictionaryIface::equal;
        if(nfeat) { setNumberOfFeatures(nfeat); }
    };

    void setAllFeatures(const NumericTableFeature &defaultFeature) DAAL_C11_OVERRIDE
    {
        if (_nfeat > 0)
        {
            _dict[0] = defaultFeature;
        }
    }

    void setNumberOfFeatures(size_t nfeat) DAAL_C11_OVERRIDE
    {
        _nfeat = nfeat;
    }
};

template <typename T, CpuType cpu>
class HomogenNumericTableCPU {};

template <CpuType cpu>
class HomogenNumericTableCPU<float, cpu> : public HomogenNumericTable<float>
{
public:
    HomogenNumericTableCPU( float *const ptr, size_t featnum, size_t obsnum )
        : HomogenNumericTable<float>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        _ptr = ptr;
        _memStatus = userAllocated;

        NumericTableFeature df;
        df.setType<float>();
        _cpuDict->setAllFeatures(df);

        setNumberOfRows( obsnum );
    }

    HomogenNumericTableCPU( size_t featnum, size_t obsnum )
        : HomogenNumericTable<float>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        setNumberOfRows( obsnum );

        NumericTableFeature df;
        df.setType<float>();
        _cpuDict->setAllFeatures(df);

        allocateDataMemory();
    }

    virtual ~HomogenNumericTableCPU() {
        delete _cpuDict;
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

private:
    NumericTableDictionary *_cpuDict;
};

template <CpuType cpu>
class HomogenNumericTableCPU<double, cpu> : public HomogenNumericTable<double>
{
public:
    HomogenNumericTableCPU( double *const ptr, size_t featnum, size_t obsnum )
        : HomogenNumericTable<double>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();

        NumericTableFeature df;
        df.setType<double>();
        _cpuDict->setAllFeatures(df);

        _ptr = ptr;
        _memStatus = userAllocated;
        setNumberOfRows( obsnum );
    }

    HomogenNumericTableCPU( size_t featnum, size_t obsnum )
        : HomogenNumericTable<double>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        setNumberOfRows( obsnum );

        NumericTableFeature df;
        df.setType<double>();
        _cpuDict->setAllFeatures(df);

        allocateDataMemory();
    }

    virtual ~HomogenNumericTableCPU() {
        delete _cpuDict;
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

private:
    NumericTableDictionary *_cpuDict;
};

template <CpuType cpu>
class HomogenNumericTableCPU<int, cpu> : public HomogenNumericTable<int>
{
public:
    HomogenNumericTableCPU( int *const ptr, size_t featnum, size_t obsnum )
        : HomogenNumericTable<int>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();

        NumericTableFeature df;
        df.setType<int>();
        _cpuDict->setAllFeatures(df);

        _ptr = ptr;
        _memStatus = userAllocated;
        setNumberOfRows( obsnum );
    }

    HomogenNumericTableCPU( size_t featnum, size_t obsnum )
        : HomogenNumericTable<int>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();

        NumericTableFeature df;
        df.setType<int>();
        _cpuDict->setAllFeatures(df);

        setNumberOfRows( obsnum );
        allocateDataMemory();
    }

    virtual ~HomogenNumericTableCPU() {
        delete _cpuDict;
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

private:
    NumericTableDictionary *_cpuDict;
};

template<typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode, typename NumericTableType>
class GetRows
{
public:
    GetRows(NumericTableType& data, size_t iStartFrom, size_t nRows) : _data(&data)
    {
        _data->getBlockOfRows(iStartFrom, nRows, mode, m_block);
    }
    GetRows(NumericTableType* data, size_t iStartFrom, size_t nRows) : _data(data)
    {
        if(_data)
            _data->getBlockOfRows(iStartFrom, nRows, mode, m_block);
    }
    GetRows() : _data(nullptr){}
    ~GetRows() { release(); }
    algorithmFPAccessType* get() { return _data ? m_block.getBlockPtr() : nullptr; }
    algorithmFPAccessType* next(size_t iStartFrom, size_t nRows)
    {
        if(!_data)
            return nullptr;
        _data->releaseBlockOfRows(m_block);
        _data->getBlockOfRows(iStartFrom, nRows, mode, m_block);
        return m_block.getBlockPtr();
    }
    algorithmFPAccessType* set(NumericTableType* data, size_t iStartFrom, size_t nRows)
    {
        release();
        if(data)
        {
            _data = data;
            _data->getBlockOfRows(iStartFrom, nRows, mode, m_block);
            return m_block.getBlockPtr();
        }
        return nullptr;
    }
    void release()
    {
        if(_data)
        {
            _data->releaseBlockOfRows(m_block);
            _data = nullptr;
        }
    }

private:
    NumericTableType* _data;
    BlockDescriptor<algorithmFPType> m_block;
};

template<typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using ReadRows = GetRows<algorithmFPType, const algorithmFPType, cpu, readOnly, NumericTableType>;

template<typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteRows = GetRows<algorithmFPType, algorithmFPType, cpu, readWrite, NumericTableType>;

template<typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteOnlyRows = GetRows<algorithmFPType, algorithmFPType, cpu, writeOnly, NumericTableType>;

template<typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode>
class GetRowsCSR
{
public:
    GetRowsCSR(CSRNumericTableIface& data, size_t iStartFrom, size_t nRows) : _data(&data)
    {
        _data->getSparseBlock(iStartFrom, nRows, mode, m_block);
    }
    GetRowsCSR(CSRNumericTableIface* data, size_t iStartFrom, size_t nRows) : _data(data)
    {
        if(_data)
            _data->getSparseBlock(iStartFrom, nRows, mode, m_block);
    }
    GetRowsCSR() : _data(nullptr){}
    ~GetRowsCSR() { release(); }
    algorithmFPAccessType* values() const { return _data ? m_block.getBlockValuesPtr() : nullptr; }
    const size_t* cols() const { return _data ? m_block.getBlockColumnIndicesPtr() : nullptr; }
    const size_t* rows() const { return _data ? m_block.getBlockRowIndicesPtr() : nullptr; }
    void next(size_t iStartFrom, size_t nRows)
    {
        if(_data)
        {
            _data->releaseSparseBlock(m_block);
            _data->getSparseBlock(iStartFrom, nRows, mode, m_block);
        }
    }
    void set(CSRNumericTableIface* data, size_t iStartFrom, size_t nRows)
    {
        release();
        if(data)
        {
            _data = data;
            _data->getSparseBlock(iStartFrom, nRows, mode, m_block);
        }
    }
    void release()
    {
        if(_data)
        {
            _data->releaseSparseBlock(m_block);
            _data = nullptr;
        }
    }

private:
    CSRNumericTableIface* _data;
    CSRBlockDescriptor<algorithmFPType> m_block;
};

template<typename algorithmFPType, CpuType cpu>
using ReadRowsCSR = GetRowsCSR<algorithmFPType, const algorithmFPType, cpu, readOnly>;

template<typename algorithmFPType, CpuType cpu>
using WriteRowsCSR = GetRowsCSR<algorithmFPType, algorithmFPType, cpu, readWrite>;

template<typename algorithmFPType, CpuType cpu>
using WriteOnlyRowsCSR = GetRowsCSR<algorithmFPType, algorithmFPType, cpu, writeOnly>;

//Simple container allocated as a pure memory block
template<CpuType cpu>
class SmartPtr
{
public:
    SmartPtr(size_t n) : _data(nullptr) { _data = (n ? daal::services::daal_malloc(n) : nullptr); }
    ~SmartPtr() { if(_data) daal::services::daal_free(_data); }
    void* get() { return _data; }
    const void* get() const { return _data; }
    void reset(size_t n)
    {
        if(_data)
        {
            daal::services::daal_free(_data);
            _data = nullptr;
        }
        if(n)
            _data = daal::services::daal_malloc(n);
    }

private:
    void* _data;
};

//Simple unique pointer container similar to std::unique_ptr
template<typename T, CpuType cpu>
class UniquePtr
{
public:
    explicit UniquePtr(T *object = nullptr) : _object(object) { }
    UniquePtr(UniquePtr<T, cpu> &&other) : _object(other.release()) { }
    ~UniquePtr() { reset(); }

    T *get() { return _object; }
    const T *get() const { return _object; }

    T &operator * () { return *_object; }
    const T &operator * () const { return *_object; }

    T *operator -> () { return _object; }
    const T *operator -> () const { return _object; }

    bool operator () () const { return _object != nullptr; }

    UniquePtr<T, cpu> &operator = (UniquePtr<T, cpu> &&other)
    {
        _object = other.release();
        return *this;
    }

    void reset(T *object = nullptr)
    {
        if (_object) { delete _object; }
        _object = object;
    }

    T *release()
    {
        T *result = _object;
        _object = nullptr;
        return result;
    }

    // Disable copy & assigment from value
    UniquePtr(const UniquePtr<T, cpu> &) = delete;
    UniquePtr<T, cpu> &operator = (const UniquePtr<T, cpu> &) = delete;

private:
    T *_object;
};

//Simple container allocated as a pure memory block using scalable malloc
template<typename T, CpuType cpu>
class TScalablePtr
{
public:
    TScalablePtr(size_t n = 0, bool isCalloc = false) : _data(nullptr) { alloc(n, isCalloc); }
    ~TScalablePtr() { destroy(); }
    T* get() { return _data; }
    const T* get() const { return _data; }

    void reset(size_t n, bool isCalloc = false)
    {
        destroy();
        alloc(n, isCalloc);
    }

private:
    void alloc(size_t n, bool isCalloc)
    {
        if (n)
        {
            _data = (isCalloc ? services::internal::service_scalable_calloc<T, cpu>(n) : services::internal::service_scalable_malloc<T, cpu>(n));
        }
    }

    void destroy()
    {
        if(_data)
        {
            services::internal::service_scalable_free<T, cpu>(_data);
            _data = nullptr;
        }
    }

    T* _data;
};


//Simple container calling constructor/destructor on its members
template<typename T, CpuType cpu>
class TArray
{
public:
    TArray(size_t n = 0) : _data(nullptr), _size(0){ alloc(n); }
    ~TArray() { destroy(); }
    T* get() { return _data; }
    const T* get() const { return _data; }
    size_t size() const { return _size; }

    void reset(size_t n)
    {
        destroy();
        alloc(n);
    }

    T &operator [] (size_t index)
    {
        return _data[index];
    }

    const T &operator [] (size_t index) const
    {
        return _data[index];
    }

private:
    void alloc(size_t n)
    {
        _data = (T*)(n ? daal::services::daal_malloc(n*sizeof(T)) : nullptr);
        if(_data)
        {
            for(size_t i = 0; i < n; ++i)
                ::new(_data + i)T;
            _size = n;
        }
    }

    void destroy()
    {
        if(_data)
        {
            for(size_t i = 0; i < _size; ++i)
                _data[i].~T();
            daal::services::daal_free(_data);
            _data = nullptr;
            _size = 0;
        }
    }

private:
    T* _data;
    size_t _size;
};


//Simple container wih initial buffer of size N, calling constructor/destructor on its members
template<typename T, size_t N, CpuType cpu>
class TNArray
{
public:
    TNArray(size_t n = 0) : _data(nullptr), _size(0){ alloc(n); }
    ~TNArray() { destroy(); }
    T* get() { return _data; }
    const T* get() const { return _data; }
    size_t size() const { return _size; }

    void reset(size_t n)
    {
        destroy();
        alloc(n);
    }

    T &operator [] (size_t index)
    {
        return _data[index];
    }

    const T &operator [] (size_t index) const
    {
        return _data[index];
    }

private:
    void alloc(size_t n)
    {
        if(n <= N)
        {
            _data = _buffer;
            _size = n;
        }
        else
        {
            _data = (T*)(n ? daal::services::daal_malloc(n*sizeof(T)) : nullptr);
            if(_data)
            {
                for(size_t i = 0; i < n; ++i)
                    ::new(_data + i)T;
                _size = n;
            }
        }
    }

    void destroy()
    {
        if(_data)
        {
            if(_data != _buffer)
            {
                for(size_t i = 0; i < _size; ++i)
                    _data[i].~T();
                daal::services::daal_free(_data);
            }
            _data = nullptr;
            _size = 0;
        }
    }

private:
    T _buffer[N];
    T* _data;
    size_t _size;
};

} // internal namespace
} // daal namespace

#endif
