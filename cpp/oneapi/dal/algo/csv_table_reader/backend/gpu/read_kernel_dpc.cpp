/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/csv_table_reader/backend/gpu/read_kernel.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/table_builder.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/memory.hpp"

#include "daal/include/data_management/data_source/file_data_source.h"
#include "daal/include/data_management/data_source/csv_feature_manager.h"

namespace oneapi::dal::csv_table_reader::backend {

read_result read_kernel_gpu::operator()(const dal::backend::context_gpu& ctx,
                                        const descriptor_base& params,
                                        const read_input& input) const {
    auto& queue = ctx.get_queue();

    using namespace daal::data_management;

    CsvDataSourceOptions csv_options =
        CsvDataSourceOptions::allocateNumericTable |
        CsvDataSourceOptions::createDictionaryFromContext |
        (params.get_parse_header() ? CsvDataSourceOptions::parseHeader : CsvDataSourceOptions::byDefault);

    FileDataSource<CSVFeatureManager> data_source(input.get_input_stream().get_file_name(), csv_options);
    data_source.getFeatureManager().setDelimiter(params.get_delimiter());
    data_source.loadDataBlock();

    auto nt = data_source.getNumericTable();

    daal::data_management::BlockDescriptor<DAAL_DATA_TYPE> block;
    const std::int64_t row_count    = nt->getNumberOfRows();
    const std::int64_t column_count = nt->getNumberOfColumns();

    nt->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
    DAAL_DATA_TYPE* data = block.getBlockPtr();

    auto arr = array<DAAL_DATA_TYPE>::empty(queue, row_count * column_count);
    dal::detail::memcpy(queue, arr.get_mutable_data(), data, sizeof(DAAL_DATA_TYPE) * row_count * column_count);

    nt->releaseBlockOfRows(block);

    return read_result{}.set_table(homogen_table_builder{}.reset(arr, row_count, column_count).build());
}

} // namespace oneapi::dal::csv_table_reader::backend
