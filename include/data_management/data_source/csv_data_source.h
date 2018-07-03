/* file: csv_data_source.h */
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
//  Implementation of the file data source class.
//--
*/

#ifndef __CSV_DATA_SOURCE_H__
#define __CSV_DATA_SOURCE_H__

#include "services/daal_memory.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @ingroup data_sources
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__FILEDATASOURCE"></a>
 *  \brief Specifies methods to access data stored in files
 *  \tparam _featureManager     FeatureManager to use to get numeric data from file strings
 */
template< typename _featureManager, typename _summaryStatisticsType = DAAL_SUMMARY_STATISTICS_TYPE >
class CsvDataSource : public DataSourceTemplate<data_management::HomogenNumericTable<DAAL_DATA_TYPE>, _summaryStatisticsType>
{
public:
    using DataSource::checkDictionary;
    using DataSource::checkNumericTable;
    using DataSource::freeNumericTable;
    using DataSource::_dict;
    using DataSource::_initialMaxRows;

    /**
     *  Typedef that stores the parser datatype
     */
    typedef _featureManager FeatureManager;

protected:
    typedef data_management::HomogenNumericTable<DAAL_DATA_TYPE> DefaultNumericTableType;

    FeatureManager featureManager;

public:
    /**
     *  Main constructor for a Data Source
     *  \param[in]  doAllocateNumericTable          Flag that specifies whether a Numeric Table
     *                                              associated with a File Data Source is allocated inside the Data Source
     *  \param[in]  doCreateDictionaryFromContext   Flag that specifies whether a Data %Dictionary
     *                                              is created from the context of the File Data Source
     *  \param[in]  initialMaxRows                  Initial value of maximum number of rows in Numeric Table allocated in loadDataBlock() method
     */
    CsvDataSource(DataSourceIface::NumericTableAllocationFlag doAllocateNumericTable    = DataSource::notAllocateNumericTable,
                  DataSourceIface::DictionaryCreationFlag doCreateDictionaryFromContext = DataSource::notDictionaryFromContext,
                  size_t initialMaxRows = 10):
        DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>(doAllocateNumericTable, doCreateDictionaryFromContext),
        _rawLineBuffer(NULL),
        _rawLineLength(0)
    {
        _rawLineBufferLen = 1024;
        _rawLineBuffer = (char *)daal::services::daal_malloc(_rawLineBufferLen);

        _contextDictFlag = false;
        _initialMaxRows  = initialMaxRows;
    }

    ~CsvDataSource()
    {
        daal::services::daal_free( _rawLineBuffer );
        DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::freeNumericTable();
    }

    /**
     *  Returns a feature manager associated with a File Data Source
     *  \return Feature manager associated with the File Data Source
     */
    FeatureManager &getFeatureManager()
    {
        return featureManager;
    }

public:
    size_t getNumericTableNumberOfColumns() DAAL_C11_OVERRIDE
    {
        return featureManager.getNumericTableNumberOfColumns();
    }

    services::Status setDictionary(DataSourceDictionary *dict) DAAL_C11_OVERRIDE
    {
        services::Status s = DataSource::setDictionary(dict);
        featureManager.setFeatureDetailsFromDictionary(dict);

        return s;
    }

    size_t loadDataBlock(NumericTable* nt) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        s = checkInputNumericTable(nt);
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        size_t maxRows = (_initialMaxRows > 0 ? _initialMaxRows : 10);
        size_t nrows = 0;
        const size_t ncols = getNumericTableNumberOfColumns();
        DataCollection tables;
        for( ;; maxRows *= 2)
        {
            NumericTablePtr ntCurrent = HomogenNumericTable<DAAL_DATA_TYPE>::create(ncols, maxRows, NumericTableIface::doAllocate, &s);
            if (!s)
            {
                this->_status.add(services::throwIfPossible(services::Status(services::ErrorNumericTableNotAllocated)));
                break;
            }
            tables.push_back(ntCurrent);
            const size_t rows = loadDataBlock(maxRows, ntCurrent.get());
            nrows += rows;
            if (rows < maxRows)
                break;
        }

        s = resetNumericTable(nt, nrows);
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        BlockDescriptor<DAAL_DATA_TYPE> blockCurrent, block;
        size_t pos = 0;
        for (size_t i = 0; i < tables.size(); i++)
        {
            NumericTable *ntCurrent = (NumericTable*)(tables[i].get());
            size_t rows = ntCurrent->getNumberOfRows();

            if(!rows)
                continue;

            ntCurrent->getBlockOfRows(0, rows, readOnly, blockCurrent);
            nt->getBlockOfRows(pos, rows, writeOnly, block);

            services::daal_memcpy_s(block.getBlockPtr(), rows * ncols * sizeof(DAAL_DATA_TYPE), blockCurrent.getBlockPtr(), rows * ncols * sizeof(DAAL_DATA_TYPE));

            ntCurrent->releaseBlockOfRows(blockCurrent);
            nt->releaseBlockOfRows(block);

            DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::combineStatistics( ntCurrent, nt, pos == 0);
            pos += rows;
        }
        return nrows;
    }

    size_t loadDataBlock(size_t maxRows, NumericTable* nt) DAAL_C11_OVERRIDE
    {
        size_t nLines = loadDataBlock(maxRows, 0, maxRows, nt);
        nt->resize( nLines );
        return nLines;
    }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows, NumericTable *nt) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        s = checkInputNumericTable(nt);
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        if (rowOffset + maxRows > fullRows)
        {
            this->_status.add(services::throwIfPossible(services::ErrorIncorrectDataRange));
            return 0;
        }

        s = resetNumericTable(nt, fullRows);
        if(!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        size_t j = 0;
        for(; j < maxRows && !iseof() ; j++ )
        {
            s = readLine();
            if(!s || !_rawLineLength)
                break;
            featureManager.parseRowIn( _rawLineBuffer, _rawLineLength, this->_dict.get(), nt, rowOffset + j );

            DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::updateStatistics( j, nt, rowOffset );
        }

        return rowOffset + j;
    }

    size_t loadDataBlock() DAAL_C11_OVERRIDE
    {
        return DataSource::loadDataBlock();
    }

    size_t loadDataBlock(size_t maxRows) DAAL_C11_OVERRIDE
    {
        return DataSource::loadDataBlock(maxRows);
    }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows) DAAL_C11_OVERRIDE
    {
        return DataSource::loadDataBlock(maxRows, rowOffset, fullRows);
    }


    services::Status createDictionaryFromContext() DAAL_C11_OVERRIDE
    {
        if(_dict)
            return services::throwIfPossible(services::Status(services::ErrorDictionaryAlreadyAvailable));

        _contextDictFlag = true;
        services::Status s;
        _dict = DataSourceDictionary::create(&s);
        if(!s) return s;

        s = readLine();
        if(!s)
        {
            this->_dict.reset();
            return services::throwIfPossible(s);
        }

        featureManager.parseRowAsDictionary( _rawLineBuffer, _rawLineLength, this->_dict.get() );
        return services::Status();
    }

    size_t getNumberOfAvailableRows() DAAL_C11_OVERRIDE
    {
        return 0;
    }

protected:
    virtual bool iseof() const = 0;
    virtual services::Status readLine() = 0;

    virtual services::Status resetNumericTable(NumericTable *nt, const size_t newSize)
    {
        services::Status s;

        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        const size_t nFeatures = getNumericTableNumberOfColumns();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
            ntDict->setFeature((*_dict)[i].ntFeature, i);

        s = DataSourceTemplate<DefaultNumericTableType, _summaryStatisticsType>::resizeNumericTableImpl(newSize, nt);
        if(!s)
        {
            return s;
        }

        nt->setNormalizationFlag(NumericTable::nonNormalized);
        return services::Status();
    }

    virtual services::Status checkInputNumericTable(const NumericTable* const nt) const
    {
        if(!nt)
        {
            return services::Status(services::ErrorNullInputNumericTable);
        }

        const NumericTable::StorageLayout layout = nt->getDataLayout();
        if (layout == NumericTable::csrArray)
        {
            return services::Status(services::ErrorIncorrectTypeOfInputNumericTable);
        }

        return services::Status();
    }

    bool enlargeBuffer()
    {
        int newRawLineBufferLen = _rawLineBufferLen * 2;
        char* newRawLineBuffer = (char *)daal::services::daal_malloc( newRawLineBufferLen );
        if(newRawLineBuffer == 0)
            return false;
        daal::services::daal_memcpy_s(newRawLineBuffer, newRawLineBufferLen, _rawLineBuffer, _rawLineBufferLen);
        daal::services::daal_free( _rawLineBuffer );
        _rawLineBuffer = newRawLineBuffer;
        _rawLineBufferLen = newRawLineBufferLen;
        return true;
    }

protected:
    char *_rawLineBuffer;
    int   _rawLineBufferLen;
    int   _rawLineLength;

    bool _contextDictFlag;
};
/** @} */
} // namespace interface1
using interface1::CsvDataSource;

}
}
#endif
