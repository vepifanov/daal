/* file: sgd_dense_momentum_impl.i */
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
//  Implementation of sgd momentum algorithm
//--
*/

#ifndef __SGD_DENSE_MOMENTUM_IMPL_I__
#define __SGD_DENSE_MOMENTUM_IMPL_I__

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_utils.h"
#include "service_rng.h"
#include "service_numeric_table.h"
#include "iterative_solver_kernel.h"
#include "threading.h"
#include <math.h>

using namespace daal::algorithms::optimization_solver::iterative_solver::internal;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{

/**
 *  \brief Kernel for SGD momentum calculation
 */
template<typename algorithmFPType, CpuType cpu>
void SGDKernel<algorithmFPType, momentum, cpu>::compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
        Parameter<momentum> *parameter, NumericTable *learningRateSequence,
        NumericTable *batchIndices, OptionalArgument *optionalArgument, OptionalArgument *optionalResult)
{
    const size_t argumentSize = inputArgument->getNumberOfRows();
    const size_t maxIterations = parameter->nIterations;
    const size_t batchSize     = parameter->batchSize;
    const double accuracyThreshold = parameter->accuracyThreshold;
    const double momentum = parameter->momentum;

    /* if maxIterations == 0, set result as start point, the number of executed iters to 0 */
    WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
    int *nProceededIterations = nIterationsBD.get();
    if(maxIterations == 0) { nProceededIterations[0] = 0; return; }

    sum_of_functions::BatchPtr function = parameter->function;

    const size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;

    SGDmomentumTask<algorithmFPType, cpu> task(batchSize,
            nTerms,
            minimum,
            batchIndices,
            this->_errors,
            optionalArgument ? NumericTable::cast(optionalArgument->get(pastUpdateVector)).get() : nullptr,
            optionalResult ? NumericTable::cast(optionalResult->get(pastUpdateVector)).get() : nullptr,
            parameter);

    task.setStartValue(inputArgument, minimum);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, task.minimimWrapper);
    function->sumOfFunctionsParameter->batchIndices = task.ntBatchIndices;

    ReadRows<int, cpu> predefinedBatchIndicesBD(batchIndices, 0, maxIterations);
    using namespace iterative_solver::internal;
    RngTask<int, cpu> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    DAAL_CHECK(batchIndices || rngTask.init(optionalArgument, nTerms, parameter->seed, sgd::rngState), ErrorMemoryAllocationFailed);

    ReadRows<algorithmFPType, cpu, NumericTable> learningRateBD(*learningRateSequence, 0, 1);
    const algorithmFPType *learningRateArray = learningRateBD.get();
    const size_t learningRateLength = learningRateSequence->getNumberOfColumns();

    size_t epoch;
    for(epoch = 0; epoch < maxIterations; epoch++)
    {
        if(task.indicesStatus == user || task.indicesStatus == random)
        {
            task.ntBatchIndices->setArray(const_cast<int *>(rngTask.get(*this->_errors, RngTask<int, cpu>::eUniformWithoutReplacement)));
        }

        function->computeNoThrow();
        if(function->getErrors()->size() != 0) {this->_errors->add(function->getErrors()->getErrors()); break;}

        NumericTable *gradient = function->getResult()->get(objective_function::gradientIdx).get();
        if(maxIterations != 1)
        {
            const algorithmFPType pointNorm = vectorNorm(minimum);
            const algorithmFPType gradientNorm = vectorNorm(gradient);
            const algorithmFPType one = 1.0;
            const algorithmFPType gradientThreshold = accuracyThreshold * daal::internal::Math<algorithmFPType, cpu>::sMax(one, pointNorm);
            if(gradientNorm < gradientThreshold) { break; }
        }

        const algorithmFPType learningRate = (learningRateLength > 1 ? learningRateArray[epoch] : learningRateArray[0]);

        task.makeStep(gradient, minimum, task.pastUpdate.get(), learningRate, momentum);
    }
    nProceededIterations[0] = (int)epoch;

    if(parameter->optionalResultRequired && !rngTask.save(optionalResult, sgd::rngState, *this->_errors))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }
}

template<typename algorithmFPType, CpuType cpu>
void SGDmomentumTask<algorithmFPType, cpu>::makeStep(
    NumericTable *gradient,
    NumericTable *minimum,
    NumericTable *pastUpdate,
    const algorithmFPType learningRate,
    const algorithmFPType momentum)
{
    processByBlocks<cpu>(minimum->getNumberOfRows(), _errors.get(),  [ = ](size_t startOffset, size_t nRowsInBlock)
    {
        WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
        algorithmFPType *workArray = workValueBD.get();
        WriteRows<algorithmFPType, cpu, NumericTable> pastUpdateBD(*pastUpdate, startOffset, nRowsInBlock);
        algorithmFPType *pastUpdateArray = pastUpdateBD.get();
        ReadRows<algorithmFPType, cpu, NumericTable> gradientBD(*gradient, startOffset, nRowsInBlock);
        const algorithmFPType *gradientArray = gradientBD.get();

        for(size_t j = 0; j < nRowsInBlock; j++)
        {
            pastUpdateArray[j] = - learningRate * gradientArray[j] + momentum * pastUpdateArray[j];
            workArray[j] = workArray[j] + pastUpdateArray[j];
        }
    });
}

template<typename algorithmFPType, CpuType cpu>
SGDmomentumTask<algorithmFPType, cpu>::~SGDmomentumTask()
{}

template<typename algorithmFPType, CpuType cpu>
void SGDmomentumTask<algorithmFPType, cpu>::setStartValue(NumericTable *inputArgument, NumericTable *minimum)
{
    processByBlocks<cpu>(minimum->getNumberOfRows(), _errors.get(),  [ = ](size_t startOffset, size_t nRowsInBlock)
    {
        WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
        algorithmFPType *workArray = workValueBD.get();
        ReadRows<algorithmFPType, cpu, NumericTable> startValueBD(*inputArgument, startOffset, nRowsInBlock);
        const algorithmFPType *startValueArray = startValueBD.get();
        if( workArray != startValueArray )
        {
            daal_memcpy_s(workArray, nRowsInBlock * sizeof(algorithmFPType), startValueArray, nRowsInBlock * sizeof(algorithmFPType));
        }
    });
}

template<typename algorithmFPType, CpuType cpu>
SGDmomentumTask<algorithmFPType, cpu>::SGDmomentumTask(
    size_t batchSize_,
    size_t nTerms_,
    NumericTable *resultTable,
    NumericTable *batchIndicesTable,
    const services::KernelErrorCollectionPtr &errors,
    NumericTable *pastUpdateInput,
    NumericTable *pastUpdateResult,
    Parameter<momentum> *parameter
) :
    batchSize(batchSize_),
    nTerms(nTerms_),
    minimimWrapper(resultTable, EmptyDeleter<NumericTable>()),
    _errors(errors),
    pastUpdate(pastUpdateResult, EmptyDeleter<NumericTable>())
{
    if(batchIndicesTable != NULL)
    {
        indicesStatus = user;
    }
    else
    {
        if(batchSize < nTerms)
        {
            indicesStatus = random;
        }
        else
        {
            indicesStatus = all;
        }
    }

    if(indicesStatus == user || indicesStatus == random)
    {
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>(new HomogenNumericTableCPU<int, cpu>(NULL, batchSize, 1));
    }
    else if(indicesStatus == all)
    {
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>();
    }

    size_t argumentSize = resultTable->getNumberOfRows();
    if(!parameter->optionalResultRequired)
    {
        SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu>> pastUpdateCpuNt =
                    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu>>(new HomogenNumericTableCPU<algorithmFPType, cpu>(1, argumentSize));
        pastUpdateCpuNt->assign(0.0);
        pastUpdate = pastUpdateCpuNt;
        return;
    }

    if(pastUpdateInput != nullptr)
    {
        if(pastUpdateInput != pastUpdate.get())
        {
            /* copy optional input ot optional result */
            processByBlocks<cpu>(argumentSize, _errors.get(),  [ = ](size_t startOffset, size_t nRowsInBlock)
            {
                WriteOnlyRows<algorithmFPType, cpu, NumericTable> pastUpdateBD(*pastUpdate, startOffset, nRowsInBlock);
                algorithmFPType *pastUpdateArray = pastUpdateBD.get();
                ReadRows<algorithmFPType, cpu, NumericTable> pastUpdateInputBD(*pastUpdateInput, startOffset, nRowsInBlock);
                const algorithmFPType *pastUpdateInputArray = pastUpdateInputBD.get();
                if( pastUpdateArray != pastUpdateInputArray )
                {
                    daal_memcpy_s(pastUpdateArray, nRowsInBlock * sizeof(algorithmFPType), pastUpdateInputArray, nRowsInBlock * sizeof(algorithmFPType));
                }
            });
        }
    }
    else /* empty optional input, set optional result to zero */
    {
        processByBlocks<cpu>(argumentSize, _errors.get(),  [ = ](size_t startOffset, size_t nRowsInBlock)
        {
            WriteOnlyRows<algorithmFPType, cpu, NumericTable> pastUpdateBD(*pastUpdate, startOffset, nRowsInBlock);
            algorithmFPType *pastUpdateArray = pastUpdateBD.get();
            for(size_t i = 0; i < nRowsInBlock; i++)
            {
                pastUpdateArray[i] = 0.0;
            }
        });
    }
}

} // namespace daal::internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
