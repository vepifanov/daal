/* file: dbscan_utils.h */
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

/*
//++
//  Common functions for DBSCAN
//--
*/

#ifndef __DBSCAN_IMPL_I__
#define __DBSCAN_IMPL_I__

#include "dbscan_types.h"

#include "threading.h"
#include "daal_defines.h"
#include "error_handling.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_kernel_math.h"
#include "service_error_handling.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::algorithms::internal;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
#define __DBSCAN_DEFAULT_QUEUE_SIZE        8
#define __DBSCAN_DEFAULT_VECTOR_SIZE       8
#define __DBSCAN_DEFAULT_NEIGHBORHOOD_SIZE 8

template <typename T, CpuType cpu>
class Queue
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_QUEUE_SIZE;

public:
    DAAL_NEW_DELETE();

    Queue() : _data(nullptr), _head(0), _tail(0), _capacity(0) {}

    ~Queue() { clear(); }

    Queue(const Queue &) = delete;
    Queue & operator=(const Queue &) = delete;

    void clear()
    {
        if (_data)
        {
            services::daal_free(_data);
            _data = nullptr;
        }
    }

    void reset() { _head = _tail = 0; }

    services::Status push(const T & value)
    {
        if (_tail >= _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }

        _data[_tail] = value;
        _tail++;

        return services::Status();
    }

    T pop()
    {
        if (_head < _tail)
        {
            const T value = _data[_head];
            _head++;
            return value;
        }

        return (T)0;
    }

    bool empty() const { return (_head == _tail); }
    size_t head() const { return _head; }
    size_t tail() const { return _tail; }

    T * getInternalPtr(size_t ind)
    {
        if (ind >= _tail)
        {
            return nullptr;
        }

        return &_data[ind];
    }

private:
    services::Status grow()
    {
        int result        = 0;
        _capacity         = (_capacity == 0 ? defaultSize : _capacity * 2);
        T * const newData = static_cast<T *>(services::internal::service_calloc<T, cpu>(_capacity * sizeof(T)));
        DAAL_CHECK_MALLOC(newData);

        if (_data != nullptr)
        {
            result = services::internal::daal_memcpy_s(newData, _tail * sizeof(T), _data, _tail * sizeof(T));
            services::daal_free(_data);
            _data = nullptr;
        }

        _data = newData;
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    T * _data;
    size_t _head;
    size_t _tail;
    size_t _capacity;
};

template <typename T, CpuType cpu>
class Vector
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_VECTOR_SIZE;

public:
    DAAL_NEW_DELETE();

    Vector() : _data(nullptr), _size(0), _capacity(0) {}

    ~Vector() { clear(); }

    Vector(const Vector &) = delete;
    Vector & operator=(const Vector &) = delete;

    void clear()
    {
        if (_data)
        {
            services::daal_free(_data);
            _data = nullptr;
        }
    }

    void reset() { _size = 0; }

    services::Status push_back(const T & value)
    {
        if (_size >= _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status);
        }

        _data[_size] = value;
        _size++;

        return services::Status();
    }

    inline T & operator[](size_t index) { return _data[index]; }

    size_t size() const { return _size; }

    T * ptr() { return _data; }

private:
    services::Status grow()
    {
        int result        = 0;
        _capacity         = (_capacity == 0 ? defaultSize : _capacity * 2);
        T * const newData = static_cast<T *>(services::internal::service_calloc<T, cpu>(_capacity * sizeof(T)));
        DAAL_CHECK_MALLOC(newData);

        if (_data != nullptr)
        {
            result = services::internal::daal_memcpy_s(newData, _size * sizeof(T), _data, _size * sizeof(T));
            services::daal_free(_data);
        }

        _data = newData;
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    T * _data;
    size_t _size;
    size_t _capacity;
};

template <typename FPType, CpuType cpu>
class Neighborhood
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_NEIGHBORHOOD_SIZE;

public:
    DAAL_NEW_DELETE();

    Neighborhood() : _values(nullptr), _capacity(0), _size(0), _weight(0) {}

    Neighborhood(const Neighborhood &) = delete;
    Neighborhood & operator=(const Neighborhood &) = delete;

    ~Neighborhood() { clear(); }

    void clear()
    {
        if (_values)
        {
            services::daal_free(_values);
            _values = nullptr;
        }
        _capacity = _size = 0;
        _weight           = 0;
    }

    void reset()
    {
        _size   = 0;
        _weight = 0;
    }

    services::Status add(const size_t & value, FPType w)
    {
        if (_size >= _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }

        _values[_size] = value;
        _size++;
        _weight += w;

        return services::Status();
    }

    void addWeight(FPType w) { _weight += w; }

    size_t get(size_t id) const { return _values[id]; }

    size_t size() const { return _size; }

    FPType weight() const { return _weight; }

private:
    services::Status grow()
    {
        int result               = 0;
        _capacity                = (_capacity == 0 ? defaultSize : _capacity * 2);
        void * ptr               = services::daal_calloc(_capacity * sizeof(size_t));
        size_t * const newValues = static_cast<size_t *>(ptr);
        DAAL_CHECK_MALLOC(newValues);

        if (_values != nullptr)
        {
            result = services::internal::daal_memcpy_s(newValues, _size * sizeof(size_t), _values, _size * sizeof(size_t));
            services::daal_free(_values);
            _values = nullptr;
        }

        _values = newValues;

        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    size_t * _values;

    size_t _capacity;
    size_t _size;

    FPType _weight;
};

template <typename FPType, CpuType cpu>
struct TlsNTask
{
    TlsNTask(size_t n) { neighs = new Neighborhood<FPType, cpu>[n]; }

    ~TlsNTask()
    {
        if (neighs)
        {
            delete[] neighs;
            neighs = nullptr;
        }
    }

    static TlsNTask * create(size_t n)
    {
        TlsNTask<FPType, cpu> * result = new TlsNTask<FPType, cpu>(n);
        if (!result)
        {
            return nullptr;
        }
        if (!result->neighs)
        {
            delete result;
            result = nullptr;
            return nullptr;
        }
        return result;
    }

    Neighborhood<FPType, cpu> * neighs;
};

template <typename FPType, CpuType cpu>
struct NTask
{
    NTask(size_t _n)
    {
        n        = _n;
        tlsNTask = new daal::tls<TlsNTask<FPType, cpu> *>([=]() -> TlsNTask<FPType, cpu> * { return TlsNTask<FPType, cpu>::create(n); });
    }

    ~NTask()
    {
        if (tlsNTask)
        {
            tlsNTask->reduce([=](TlsNTask<FPType, cpu> * tt) -> void {
                delete tt;
                tt = nullptr;
            });
            delete tlsNTask;
        }
    }

    static services::SharedPtr<NTask<FPType, cpu> > create(size_t n)
    {
        services::SharedPtr<NTask<FPType, cpu> > result(new NTask<FPType, cpu>(n));
        if (result.get() && (!result->tlsNTask))
        {
            result.reset();
        }
        return result;
    }

    size_t n;
    daal::tls<TlsNTask<FPType, cpu> *> * tlsNTask;
};

template <Method, typename FPType, CpuType cpu>
class NeighborhoodEngine
{
public:
    NeighborhoodEngine(const NumericTable * inTable, const NumericTable * outTable, const NumericTable * weights, FPType eps, FPType p);

    services::Status queryFull(Neighborhood<FPType, cpu> * neighs, bool doReset = false);

    services::Status query(size_t * indices, size_t n, Neighborhood<FPType, cpu> * neighs, bool doReset = false);
};

template <typename FPType, CpuType cpu>
class NeighborhoodEngine<defaultDense, FPType, cpu>
{
    DAAL_NEW_DELETE();

public:
    NeighborhoodEngine(const NumericTable * inTable, const NumericTable * outTable, const NumericTable * weights, FPType eps, FPType p)
        : _inTable(inTable), _outTable(outTable), _weights(weights), _eps(eps), _p(p)
    {}

    ~NeighborhoodEngine() {}

    NeighborhoodEngine(const NeighborhoodEngine &) = delete;
    NeighborhoodEngine & operator=(const NeighborhoodEngine &) = delete;

    services::Status queryFull(Neighborhood<FPType, cpu> * neighs, bool doReset = false)
    {
        SafeStatus safeStat;
        services::Status s;

        const size_t inRows  = _inTable->getNumberOfRows();
        const size_t outRows = _outTable->getNumberOfRows();

        if (outRows == 0)
        {
            return s;
        }

        const size_t dim    = _inTable->getNumberOfColumns();
        const size_t outDim = _outTable->getNumberOfColumns();
        DAAL_ASSERT(outDim >= dim);

        FPType epsP = Math<FPType, cpu>::sPowx(_eps, _p);

        size_t inBlockSize = 256;
        size_t nInBlocks   = inRows / inBlockSize + (inRows % inBlockSize > 0);

        size_t outBlockSize = 256;
        size_t nOutBlocks   = outRows / outBlockSize + (outRows % outBlockSize > 0);

        daal::threader_for(nInBlocks, nInBlocks, [&](size_t inBlock) {
            size_t i1    = inBlock * inBlockSize;
            size_t i2    = (inBlock + 1 == nInBlocks ? inRows : i1 + inBlockSize);
            size_t iSize = i2 - i1;

            ReadRows<FPType, cpu> inDataRows(const_cast<NumericTable *>(_inTable), i1, i2 - i1);
            DAAL_CHECK_BLOCK_STATUS_THR(inDataRows);
            const FPType * const inData = inDataRows.get();

            if (doReset)
            {
                for (size_t i = 0; i < iSize; i++)
                {
                    neighs[i + i1].reset();
                }
            }

            for (size_t outBlock = 0; outBlock < nOutBlocks; outBlock++)
            {
                size_t j1    = outBlock * outBlockSize;
                size_t j2    = (outBlock + 1 == nOutBlocks ? outRows : j1 + outBlockSize);
                size_t jSize = j2 - j1;

                ReadRows<FPType, cpu> outDataRows(const_cast<NumericTable *>(_outTable), j1, j2 - j1);
                DAAL_CHECK_BLOCK_STATUS_THR(outDataRows);
                const FPType * const outData = outDataRows.get();

                ReadRows<FPType, cpu> weightsRows;
                if (_weights)
                {
                    weightsRows.set(const_cast<NumericTable *>(_weights), j1, j2 - j1);
                    DAAL_CHECK_BLOCK_STATUS_THR(weightsRows);
                }
                const FPType * const weights = weightsRows.get();

                for (size_t i = 0; i < iSize; i++)
                {
                    for (size_t j = 0; j < jSize; j++)
                    {
                        FPType dist = distancePow2<FPType, cpu>(&inData[i * dim], &outData[j * outDim], dim);
                        if (dist <= epsP)
                        {
                            DAAL_CHECK_STATUS_THR(neighs[i + i1].add(j + j1, (weights ? weights[j] : (FPType)1.0)));
                        }
                    }
                }
            }
        });
        s = safeStat.detach();

        return s;
    }

    services::Status query(size_t * indices, size_t n, Neighborhood<FPType, cpu> * neighs, bool doReset = false)
    {
        SafeStatus safeStat;
        services::Status s;

        const size_t outRows = _outTable->getNumberOfRows();

        if (outRows == 0)
        {
            return s;
        }

        const size_t dim    = _inTable->getNumberOfColumns();
        const size_t outDim = _outTable->getNumberOfColumns();
        DAAL_ASSERT(outDim >= dim);

        services::SharedPtr<NTask<FPType, cpu> > nTask = NTask<FPType, cpu>::create(n);
        DAAL_CHECK_MALLOC(nTask.get());

        TArray<ReadRows<FPType, cpu>, cpu> queryRows(n);
        DAAL_CHECK_MALLOC(queryRows.get());

        for (size_t i = 0; i < n; i++)
        {
            queryRows[i].set(const_cast<NumericTable *>(_inTable), indices[i], 1);
            DAAL_CHECK_BLOCK_STATUS(queryRows[i]);
        }

        if (doReset)
        {
            for (size_t i = 0; i < n; i++)
            {
                neighs[i].reset();
            }
        }

        FPType epsP = Math<FPType, cpu>::sPowx(_eps, _p);

        size_t outBlockSize = 256;
        size_t nOutBlocks   = outRows / outBlockSize + (outRows % outBlockSize > 0);

        daal::threader_for(nOutBlocks, nOutBlocks, [&](size_t outBlock) {
            TlsNTask<FPType, cpu> * tlsNTask = nTask->tlsNTask->local();
            DAAL_CHECK_MALLOC_THR(tlsNTask);

            Neighborhood<FPType, cpu> * localNeighs = tlsNTask->neighs;
            DAAL_CHECK_MALLOC_THR(localNeighs);

            size_t j1    = outBlock * outBlockSize;
            size_t j2    = (outBlock + 1 == nOutBlocks ? outRows : j1 + outBlockSize);
            size_t jSize = j2 - j1;

            ReadRows<FPType, cpu> outDataRows(const_cast<NumericTable *>(_outTable), j1, j2 - j1);
            DAAL_CHECK_BLOCK_STATUS_THR(outDataRows);
            const FPType * const outData = outDataRows.get();

            ReadRows<FPType, cpu> weightsRows;
            if (_weights)
            {
                weightsRows.set(const_cast<NumericTable *>(_weights), j1, j2 - j1);
                DAAL_CHECK_BLOCK_STATUS_THR(weightsRows);
            }
            const FPType * const weights = weightsRows.get();

            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < jSize; j++)
                {
                    FPType dist = distancePow2<FPType, cpu>(queryRows[i].get(), &outData[j * outDim], dim);
                    if (dist <= epsP)
                    {
                        DAAL_CHECK_STATUS_THR(localNeighs[i].add(j + j1, (weights ? weights[j] : (FPType)1.0)));
                    }
                }
            }
        });

        s = safeStat.detach();
        DAAL_CHECK_STATUS_VAR(s);

        nTask->tlsNTask->reduce([&](TlsNTask<FPType, cpu> * tt) -> void {
            Neighborhood<FPType, cpu> * localNeighs = tt->neighs;
            if (localNeighs)
            {
                for (size_t i = 0; i < n; i++)
                {
                    size_t localSize = localNeighs[i].size();
                    for (size_t j = 0; j < localSize; j++)
                    {
                        neighs[i].add(localNeighs[i].get(j), 0);
                    }
                    neighs[i].addWeight(localNeighs[i].weight());
                }
            }
        });

        return s;
    }

private:
    const NumericTable * _inTable;
    const NumericTable * _outTable;
    const NumericTable * _weights;

    FPType _eps;
    FPType _p;
};

template <typename FPType, CpuType cpu>
FPType findKthStatistic(FPType * values, size_t nElements, size_t k)
{
    if (nElements == 0)
    {
        return (FPType)0;
    }

    if (k >= nElements)
    {
        k = nElements - 1;
    }

    size_t l = 0;
    size_t r = nElements - 1;
    while (l < r)
    {
        FPType med = values[k];
        int i      = l;
        int j      = r;
        while (i <= j)
        {
            while (values[i] < med)
            {
                i++;
            }
            while (med < values[j])
            {
                j--;
            }
            if (i <= j)
            {
                swap<cpu, FPType>(values[i], values[j]);
                i++;
                j--;
            }
        }
        if (j < k)
        {
            l = i;
        }
        if (k < i)
        {
            r = j;
        }
    }

    return values[k];
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
