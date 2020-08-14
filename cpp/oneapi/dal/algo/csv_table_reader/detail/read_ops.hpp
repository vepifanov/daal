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

#pragma once

#include "oneapi/dal/algo/csv_table_reader/read_types.hpp"
#include "oneapi/dal/data/accessor.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::csv_table_reader::detail {

template <typename Context, typename... Options>
struct ONEAPI_DAL_EXPORT read_ops_dispatcher {
    read_result operator()(const Context&, const descriptor_base&, const read_input&) const;
};

template <typename Descriptor>
struct read_ops {
    using input_t           = read_input;
    using result_t          = read_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const read_input& input) const {
        check_preconditions(desc, input);
        const auto result = read_ops_dispatcher<Context>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace oneapi::dal::csv_table_reader::detail
