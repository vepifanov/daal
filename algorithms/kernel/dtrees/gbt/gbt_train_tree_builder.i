/* file: gbt_train_tree_builder.i */
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
//  Implementation of auxiliary functions for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_TREE_BUILDER_I__
#define __GBT_TRAIN_TREE_BUILDER_I__

#include "dtrees_model_impl.h"
#include "dtrees_train_data_helper.i"
#include "dtrees_predict_dense_default_impl.i"
#include "gbt_train_aux.i"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::internal;

template<typename algorithmFPType, CpuType cpu>
class TreeBuilder : public TreeBuilderBase
{
public:
    typedef TrainBatchTaskBaseXBoost<algorithmFPType, cpu> CommonCtx;
    using MemHelperType = MemHelperBase<algorithmFPType, cpu>;
    typedef typename CommonCtx::DataHelperType DataHelperType;

    typedef gh<algorithmFPType, cpu> ghType;
    typedef SplitJob<algorithmFPType, cpu> SplitJobType;
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    typedef SplitData<algorithmFPType, ImpurityType> SplitDataType;

    struct SplitTask : public SplitJobType
    {
        typedef TreeBuilder<algorithmFPType, cpu> Task;
        typedef SplitJobType super;

        SplitTask(const SplitTask& o) : super(o), _task(o._task){}
        SplitTask(Task& task, size_t _iStart, size_t _n, size_t _level, const ImpurityType& _imp, NodeType::Base*& _res) :
            super(_iStart, _n, _level, _imp, _res), _task(task){}
        Task& _task;
        void operator()()
        {
            _task._ctx._nParallelNodes.inc();
            _task.buildSplit(*this);
            _task._ctx._nParallelNodes.dec();
        }
    };

    class BestSplit
    {
    public:
        BestSplit(SplitDataType& split, Mutex* mt) :
            _split(split), _mt(mt), _iIndexedFeatureSplitValue(-1), _iFeature(-1){}
        void safeGetData(algorithmFPType& impDec, int& iFeature)
        {
            if(_mt)
            {
                _mt->lock();
                impDec = impurityDecrease();
                iFeature = _iFeature;
                _mt->unlock();
            }
            else
            {
                impDec = impurityDecrease();
                iFeature = _iFeature;
            }
        }
        void update(const SplitDataType& split, int iIndexedFeatureSplitValue, int iFeature)
        {
            if(_mt)
            {
                _mt->lock();
                updateImpl(split, iIndexedFeatureSplitValue, iFeature);
                _mt->unlock();
            }
            else
                updateImpl(split, iIndexedFeatureSplitValue, iFeature);
        }

        void update(const SplitDataType& split, int iFeature, IndexType* bestSplitIdx, const IndexType* aIdx, size_t n)
        {
            if(_mt)
            {
                _mt->lock();
                if(updateImpl(split, -1, iFeature))
                    tmemcpy<IndexType, cpu>(bestSplitIdx, aIdx, n);
                _mt->unlock();
            }
            else
            {
                if(updateImpl(split, -1, iFeature))
                    tmemcpy<IndexType, cpu>(bestSplitIdx, aIdx, n);
            }
        }

        int iIndexedFeatureSplitValue() const { return _iIndexedFeatureSplitValue; }
        int iFeature() const { return _iFeature; }
        bool isThreadedMode() const { return _mt != nullptr; }

    private:
        algorithmFPType impurityDecrease() const { return _split.impurityDecrease; }
        bool updateImpl(const SplitDataType& split, int iIndexedFeatureSplitValue, int iFeature)
        {
            if(split.impurityDecrease < impurityDecrease())
                return false;

            if(split.impurityDecrease == impurityDecrease())
            {
                if(_iFeature < (int)iFeature) //deterministic way, let the split be the same as in sequential case
                    return false;
            }
            _iFeature = (int)iFeature;
            split.copyTo(_split);
            _iIndexedFeatureSplitValue = iIndexedFeatureSplitValue;
            return true;
        }

    private:
        SplitDataType& _split;
        Mutex* _mt;
        volatile int _iIndexedFeatureSplitValue;
        volatile int _iFeature;
    };

    TreeBuilder(CommonCtx& ctx) : _ctx(ctx){}
    ~TreeBuilder()
    {
        delete _memHelper;
        delete _taskGroup;
    }

    bool isInitialized() const { return !!_aBestSplitIdxBuf.get(); }
    virtual services::Status run(dtrees::internal::DecisionTreeTable*& pRes, size_t iTree) DAAL_C11_OVERRIDE;
    virtual services::Status init() DAAL_C11_OVERRIDE
    {
        _aBestSplitIdxBuf.reset(_ctx.nSamples());
        _aSample.reset(_ctx.nSamples());
        DAAL_CHECK_MALLOC(_aBestSplitIdxBuf.get() && _aSample.get());
        DAAL_CHECK_MALLOC(initMemHelper());
        if(_ctx.isParallelNodes() && !_taskGroup)
            DAAL_CHECK_MALLOC((_taskGroup = new daal::task_group()));
        return services::Status();
    }
    daal::task_group* taskGroup() { return _taskGroup; }

protected:
    bool initMemHelper();
    //find features to check in the current split node
    const IndexType* chooseFeatures()
    {
        if(_ctx.nFeatures() == _ctx.nFeaturesPerNode())
            return nullptr;
        IndexType* featureSample = _memHelper->getFeatureSampleBuf();
        _ctx.chooseFeatures(featureSample);
        return featureSample;
    }
    void buildNode(size_t iStart, size_t n, size_t level, const ImpurityType& imp, NodeType::Base*& res);
    IndexType* bestSplitIdxBuf() const { return _aBestSplitIdxBuf.get(); }
    NodeType::Base* buildRoot(size_t iTree)
    {
        _iTree = iTree;
        const size_t nSamples = _ctx.nSamples();
        auto aSample = _aSample.get();
        for(size_t i = 0; i < nSamples; ++i)
            aSample[i] = i;

        ImpurityType imp;
        getInitialImpurity(imp);
        typename NodeType::Base* res = buildLeaf(0, nSamples, 0, imp);
        if(res)
            return res;
        SplitJobType job(0, nSamples, 0, imp, res);
        buildSplit(job);
        if(taskGroup())
            taskGroup()->wait();
        return res;
    }

    void getInitialImpurity(ImpurityType& val)
    {
        const size_t nSamples = _ctx.nSamples();
        const ghType* pgh = _ctx.grad(this->_iTree);
        auto& G = val.g;
        auto& H = val.h;
        G = H = 0;
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nSamples; ++i)
        {
            G += pgh[i].g;
            H += pgh[i].h;
        }
    }

    void calcImpurity(const IndexType* aIdx, size_t n, ImpurityType& imp) const //todo: tree?
    {
        DAAL_ASSERT(n);
        const ghType* pgh = _ctx.grad(this->_iTree);
        imp = pgh[aIdx[0]];
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 1; i < n; ++i)
        {
            imp.g += pgh[aIdx[i]].g;
            imp.h += pgh[aIdx[i]].h;
        }
    }

    NodeType::Base* buildLeaf(size_t iStart, size_t n, size_t level, const ImpurityType& imp)
    {
        return _ctx.terminateCriteria(n, level, imp) ? makeLeaf(_aSample.get() + iStart, n, imp) : nullptr;
    }

    typename NodeType::Leaf* makeLeaf(const IndexType* idx, size_t n, const ImpurityType& imp)
    {
        typename NodeType::Leaf* pNode = nullptr;
        if(_ctx.isThreaded())
        {
            _mtAlloc.lock();
            pNode = _tree.allocator().allocLeaf();
            _mtAlloc.unlock();
        }
        else
            pNode = _tree.allocator().allocLeaf();
        pNode->response = _ctx.computeLeafWeightUpdateF(idx, n, imp, _iTree);
        return pNode;
    }
    typename NodeType::Split* makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered);
    void buildSplit(SplitJobType& job);
    bool findBestSplit(SplitJobType& job, SplitDataType& split, IndexType& iFeature);
    void findBestSplitImpl(const SplitJobType& job, SplitDataType& split, IndexType& iFeature, int& idxFeatureValueBestSplit, bool& bCopyToIdx);
    void findSplitOneFeature(const IndexType* featureSample, size_t iFeatureInSample, const SplitJobType& job, BestSplit& bestSplit);

    int findBestSplitFeatIndexed(IndexType iFeature, const IndexType* aIdx, const SplitJobType& job, SplitDataType& split, bool bUpdateWhenTie) const;

    bool simpleSplit(SplitJobType& job, SplitDataType& split, IndexType& iFeature);

    void finalizeBestSplitFeatIndexed(const IndexType* aIdx, size_t n,
        SplitDataType& bestSplit, IndexType iFeature, size_t idxFeatureValueBestSplit, IndexType* bestSplitIdx) const;

    bool findBestSplitFeatSorted(const algorithmFPType* featureVal, const IndexType* aIdx,
        const SplitJobType& job, SplitDataType& split, bool bUpdateWhenTie) const
    {
        return split.featureUnordered ? findBestSplitFeatSortedCategorical(featureVal, aIdx, job, split, bUpdateWhenTie) :
            findBestSplitFeatSortedOrdered(featureVal, aIdx, job, split, bUpdateWhenTie);
    }

    bool findBestSplitFeatSortedOrdered(const algorithmFPType* featureVal,
        const IndexType* aIdx, const SplitJobType& job, SplitDataType& split, bool bUpdateWhenTie) const;

    bool findBestSplitFeatSortedCategorical(const algorithmFPType* featureVal,
        const IndexType* aIdx, const SplitJobType& job, SplitDataType& split, bool bUpdateWhenTie) const;

protected:
    CommonCtx& _ctx;
    size_t _iTree = 0;
    TreeType _tree;
    daal::Mutex _mtAlloc;
    typedef dtrees::internal::TVector<IndexType, cpu> IndexTypeArray;
    mutable IndexTypeArray _aBestSplitIdxBuf;
    mutable IndexTypeArray _aSample;
    MemHelperType* _memHelper = nullptr;
    daal::task_group* _taskGroup = nullptr;
};

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilder<algorithmFPType, cpu>::initMemHelper()
{
    auto featuresSampleBufSize = 0; //do not allocate if not required
    const auto nFeat = _ctx.nFeatures();
    if(nFeat != _ctx.nFeaturesPerNode())
    {
        if(_ctx.nFeaturesPerNode()*_ctx.nFeaturesPerNode() < 2 * nFeat)
            featuresSampleBufSize = 2 * _ctx.nFeaturesPerNode();
        else
            featuresSampleBufSize = nFeat;
    }
    if(_ctx.isThreaded())
        _memHelper = new MemHelperThr<algorithmFPType, cpu>(featuresSampleBufSize);
    else
        _memHelper = new MemHelperSeq<algorithmFPType, cpu>(featuresSampleBufSize,
        _ctx.par().memorySavingMode ? 0 : _ctx.dataHelper().indexedFeatures().maxNumIndices(),
        _ctx.nSamples()); //TODO
    return _memHelper && _memHelper->init();
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilder<algorithmFPType, cpu>::buildNode(
    size_t iStart, size_t n, size_t level, const ImpurityType& imp, NodeType::Base*&res)
{
    if(taskGroup())
    {
        SplitTask job(*this, iStart, n, level, imp, res);
        taskGroup()->run(job);
    }
    else
    {
        SplitJobType job(iStart, n, level, imp, res);
        buildSplit(job);
    }
}

template <typename algorithmFPType, CpuType cpu>
typename TreeBuilder<algorithmFPType, cpu>::NodeType::Split*
TreeBuilder<algorithmFPType, cpu>::makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered)
{
    typename NodeType::Split* pNode = nullptr;
    if(_ctx.isThreaded())
    {
        _mtAlloc.lock();
        pNode = _tree.allocator().allocSplit();
        _mtAlloc.unlock();
    }
    else
        pNode = _tree.allocator().allocSplit();
    pNode->set(iFeature, featureValue, bUnordered);
    return pNode;
}

template <typename algorithmFPType, CpuType cpu>
services::Status TreeBuilder<algorithmFPType, cpu>::run(dtrees::internal::DecisionTreeTable*& pRes, size_t iTree)
{
    _tree.destroy();
    typename NodeType::Base* nd = buildRoot(iTree);
    DAAL_CHECK_MALLOC(nd);
    _tree.reset(nd, false); //bUnorderedFeaturesUsed - TODO?
    pRes = gbt::internal::ModelImpl::treeToTable(_tree);
    if(_ctx.isBagging() && _tree.top())
        _ctx.updateOOB(iTree, _tree);
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilder<algorithmFPType, cpu>::buildSplit(SplitJobType& job)
{
    SplitDataType split;
    IndexType iFeature;
    if(findBestSplit(job, split, iFeature))
    {
        typename NodeType::Split* res = makeSplit(iFeature, split.featureValue, split.featureUnordered);
        if(res)
        {
            job.res = res;
            res->kid[0] = buildLeaf(job.iStart, split.nLeft, job.level + 1, split.left);
            ImpurityType impRight; //statistics for the right part of the split
            //actually it is equal to job.imp - split.left, but 'imp' contains roundoff errors.
            //calculate right part directly to avoid propagation of the errors
            calcImpurity(_aSample.get() + job.iStart + split.nLeft, job.n - split.nLeft, impRight);
            res->kid[1] = buildLeaf(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, impRight);
            if(res->kid[0])
            {
                if(res->kid[1])
                    return; //all done
                SplitJobType right(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, impRight, res->kid[1]);
                buildSplit(right); //by this thread, no new job
            }
            else if(res->kid[1])
            {
                SplitJobType left(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                buildSplit(left); //by this thread, no new job
            }
            else
            {
                //one kid can be a new job, the left one, if there are available threads
                if(_ctx.numAvailableThreads())
                    buildNode(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                else
                {
                    SplitJobType left(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                    buildSplit(left); //by this thread, no new job
                }
                //and another kid is processed in the same thread
                SplitJobType right(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, impRight, res->kid[1]);
                buildSplit(right); //by this thread, no new job
            }
            return;
        }
    }
    job.res = makeLeaf(_aSample.get() + job.iStart, job.n, job.imp);
}

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilder<algorithmFPType, cpu>::findBestSplit(SplitJobType& job,
    SplitDataType& bestSplit, IndexType& iFeature)
{
    if(job.n == 2)
    {
        DAAL_ASSERT(_ctx.par().minObservationsInLeafNode == 1);
        return simpleSplit(job, bestSplit, iFeature);
    }

    int idxFeatureValueBestSplit = -1; //when sorted feature is used
    bool bCopyToIdx = true;
    findBestSplitImpl(job, bestSplit, iFeature, idxFeatureValueBestSplit, bCopyToIdx);
    if(iFeature < 0)
        return false;
    IndexType* bestSplitIdx = bestSplitIdxBuf() + job.iStart;
    IndexType* aIdx = _aSample.get() + job.iStart;
    if(idxFeatureValueBestSplit >= 0)
    {
        //indexed feature was used
        //calculate impurity (??) and get split to bestSplitIdx
        finalizeBestSplitFeatIndexed(aIdx, job.n, bestSplit, iFeature, idxFeatureValueBestSplit, bestSplitIdx);
        bCopyToIdx = true;
    }
    else if(bestSplit.featureUnordered)
    {
        if(bestSplit.iStart)
        {
            DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= job.n);
            tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx + bestSplit.iStart, bestSplit.nLeft);
            aIdx += bestSplit.nLeft;
            tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, bestSplit.iStart);
            aIdx += bestSplit.iStart;
            bestSplitIdx += bestSplit.iStart + bestSplit.nLeft;
            if(job.n > (bestSplit.iStart + bestSplit.nLeft))
                tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, job.n - bestSplit.iStart - bestSplit.nLeft);
            bCopyToIdx = false;//done
        }
    }
    if(bCopyToIdx)
        tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, job.n);
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilder<algorithmFPType, cpu>::simpleSplit(SplitJobType& job,
    SplitDataType& split, IndexType& iFeature)
{
    algorithmFPType featBuf[2];
    IndexType* aIdx = _aSample.get() + job.iStart;
    for(size_t i = 0; i < _ctx.nFeatures(); ++i)
    {
        _ctx.featureValuesToBuf(i, featBuf, aIdx, 2);
        if(featBuf[1] - featBuf[0] <= _ctx.accuracy()) //all values of the feature are the same
            continue;
        split.featureValue = featBuf[0];
        split.nLeft = 1;
        split.iStart = 0;
        _ctx.simpleSplit(_iTree, featBuf, aIdx, split.left);
        split.impurityDecrease = job.imp.value(_ctx.par().lambda);
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilder<algorithmFPType, cpu>::finalizeBestSplitFeatIndexed(const IndexType* aIdx, size_t n,
    SplitDataType& split, IndexType iFeature, size_t idxFeatureValueBestSplit, IndexType* bestSplitIdx) const
{
    DAAL_ASSERT(split.nLeft > 0);
    IndexType* bestSplitIdxRight = bestSplitIdx + split.nLeft;
    const int iRowSplitVal = doPartitionIdx<IndexType, typename IndexedFeatures::IndexType, size_t, cpu>(
        n, aIdx, _ctx.aSampleToF(),
        _ctx.dataHelper().indexedFeatures().data(iFeature), split.featureUnordered,
        idxFeatureValueBestSplit,
        bestSplitIdxRight, bestSplitIdx,
        split.nLeft);

    DAAL_ASSERT(iRowSplitVal >= 0);
    split.iStart = 0;
    if(_ctx.dataHelper().indexedFeatures().isBinned(iFeature))
        split.featureValue = (algorithmFPType)_ctx.dataHelper().indexedFeatures().binRightBorder(iFeature, idxFeatureValueBestSplit);
    else
        split.featureValue = _ctx.dataHelper().getValue(iFeature, iRowSplitVal);
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilder<algorithmFPType, cpu>::findSplitOneFeature(
    const IndexType* featureSample, size_t iFeatureInSample, const SplitJobType& job, BestSplit& bestSplit)
{
    const float qMax = 0.02; //min fracture of observations to be handled as indexed feature values
    const IndexType iFeature = featureSample ? featureSample[iFeatureInSample] : (IndexType)iFeatureInSample;
    const bool bIndexedMode = (!_ctx.par().memorySavingMode) &&
        (float(job.n) > qMax*float(_ctx.dataHelper().indexedFeatures().numIndices(iFeature)));

    if(bIndexedMode)
    {
        const IndexType* aIdx = _aSample.get() + job.iStart;
        if(!_ctx.dataHelper().hasDiffFeatureValues(iFeature, aIdx, job.n))
            return;//all values of the feature are the same
        //use best split estimation when searching on iFeature
        algorithmFPType bestImpDec;
        int iBestFeat;
        bestSplit.safeGetData(bestImpDec, iBestFeat);
        SplitDataType split(bestImpDec, _ctx.featTypes().isUnordered(iFeature));
        //index of best feature value in the array of sorted feature values
        const int idxFeatureValue = findBestSplitFeatIndexed(iFeature, aIdx, job, split, iBestFeat < 0 || iBestFeat > iFeature);
        if(idxFeatureValue < 0)
            return;
        bestSplit.update(split, idxFeatureValue, iFeature);
    }
    else
    {
        IndexType* aIdx = _aSample.get() + job.iStart;
        const bool bThreaded = bestSplit.isThreadedMode();
        IndexType* bestSplitIdx = bestSplitIdxBuf() + job.iStart;
        auto aFeatBuf = _memHelper->getFeatureValueBuf(job.n); //TODO?
        typename MemHelperType::IndexTypeVector* aFeatIdxBuf = nullptr;
        if(bThreaded)
        {
            //get a local index, since it is used by parallel threads
            aFeatIdxBuf = _memHelper->getSortedFeatureIdxBuf(job.n);
            tmemcpy<IndexType, cpu>(aFeatIdxBuf->get(), aIdx, job.n);
            aIdx = aFeatIdxBuf->get();
        }
        algorithmFPType* featBuf = aFeatBuf->get();
        _ctx.featureValuesToBuf(iFeature, featBuf, aIdx, job.n);
        if(featBuf[job.n - 1] - featBuf[0] <= _ctx.accuracy()) //all values of the feature are the same
        {
            _memHelper->releaseFeatureValueBuf(aFeatBuf);
            if(aFeatIdxBuf)
                _memHelper->releaseSortedFeatureIdxBuf(aFeatIdxBuf);
            return;
        }
        //use best split estimation when searching on iFeature
        algorithmFPType bestImpDec;
        int iBestFeat;
        bestSplit.safeGetData(bestImpDec, iBestFeat);
        SplitDataType split(bestImpDec, _ctx.featTypes().isUnordered(iFeature));
        bool bFound = findBestSplitFeatSorted(featBuf, aIdx, job, split, iBestFeat < 0 || iBestFeat > iFeature);
        _memHelper->releaseFeatureValueBuf(aFeatBuf);
        if(bFound)
        {
            DAAL_ASSERT(split.iStart < job.n);
            DAAL_ASSERT(split.iStart + split.nLeft <= job.n);
            if(split.featureUnordered || bThreaded ||
                (featureSample ? (iFeature != featureSample[_ctx.nFeaturesPerNode() - 1]) : (iFeature + 1 < _ctx.nFeaturesPerNode()))) //not a last feature
                bestSplit.update(split, iFeature, bestSplitIdx, aIdx, job.n);
            else
                bestSplit.update(split, -1, iFeature);
        }
        if(aFeatIdxBuf)
            _memHelper->releaseSortedFeatureIdxBuf(aFeatIdxBuf);
    }
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilder<algorithmFPType, cpu>::findBestSplitImpl(const SplitJobType& job,
    SplitDataType& split, IndexType& iFeature, int& idxFeatureValueBestSplit, bool& bCopyToIdx)
{
    const IndexType* featureSample = chooseFeatures();
    iFeature = -1;
    bCopyToIdx = true;
    if(_ctx.isParallelFeatures() && _ctx.numAvailableThreads())
    {
        daal::Mutex mtBestSplit;
        BestSplit bestSplit(split, &mtBestSplit);
        daal::threader_for(_ctx.nFeaturesPerNode(), _ctx.nFeaturesPerNode(), [&](size_t i)
        {
            findSplitOneFeature(featureSample, i, job, bestSplit);
        });
        idxFeatureValueBestSplit = bestSplit.iIndexedFeatureSplitValue();
        iFeature = bestSplit.iFeature();
    }
    else
    {
        BestSplit bestSplit(split, nullptr);
        for(size_t i = 0; i < _ctx.nFeaturesPerNode(); ++i)
        {
            findSplitOneFeature(featureSample, i, job, bestSplit);
        }
        idxFeatureValueBestSplit = bestSplit.iIndexedFeatureSplitValue();
        iFeature = bestSplit.iFeature();
        if((iFeature >= 0) && (idxFeatureValueBestSplit < 0) && !split.featureUnordered)
        {
            //in sequential mode, if iBestSplit is the last considered feature then aIdx already contains the best split, no need to copy
            if(featureSample ? (iFeature == featureSample[_ctx.nFeaturesPerNode() - 1]) : (iFeature + 1 == _ctx.nFeaturesPerNode())) //last feature
                bCopyToIdx = false;
        }
    }
    if(featureSample)
        _memHelper->releaseFeatureSampleBuf(const_cast<IndexType*>(featureSample));

    if(iFeature < 0)
        return; //not found
    //now calculate full impurity decrease
    split.impurityDecrease -= job.imp.value(_ctx.par().lambda);
    if(split.impurityDecrease < _ctx.par().minSplitLoss)
        iFeature = -1; //not found
}

template<typename algorithmFPType, CpuType cpu>
int TreeBuilder<algorithmFPType, cpu>::findBestSplitFeatIndexed(IndexType iFeature, const IndexType* aIdx,
    const SplitJobType& job, SplitDataType& split, bool bUpdateWhenTie) const
{
    const size_t nDiffFeatMax = _ctx.dataHelper().indexedFeatures().numIndices(iFeature);
    auto aGHSumBuf = _memHelper->getGHSumBuf(nDiffFeatMax); //sums of gradients per each value of the indexed feature
    DAAL_ASSERT(aGHSumBuf); //TODO: return status
    if(!aGHSumBuf)
        return -1;

    auto aGHSum = aGHSumBuf->get();
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(size_t i = 0; i < nDiffFeatMax; ++i)
    {
        aGHSum[i].n = 0;
        aGHSum[i].g = algorithmFPType(0);
        aGHSum[i].h = algorithmFPType(0);
    }

    const size_t n = job.n;
    algorithmFPType gTotal = 0; //total sum of g in the set being split
    algorithmFPType hTotal = 0; //total sum of h in the set being split
    {
        const IndexedFeatures::IndexType* indexedFeature = _ctx.dataHelper().indexedFeatures().data(iFeature);
        const ghType* pgh = _ctx.grad(this->_iTree);
        if(_ctx.aSampleToF())
        {
            const IndexType* aSampleToF = _ctx.aSampleToF();
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < n; ++i)
            {
                const IndexType iSample = aIdx[i];
                    const IndexType iRow = aSampleToF[iSample];
                    const typename IndexedFeatures::IndexType idx = indexedFeature[iRow];
                auto& sum = aGHSum[idx];
                sum.n++;
                sum.g += pgh[iSample].g;
                sum.h += pgh[iSample].h;
                gTotal += pgh[iSample].g;
                hTotal += pgh[iSample].h;
            }
        }
        else
        {
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < n; ++i)
            {
                const IndexType iSample = aIdx[i];
                const typename IndexedFeatures::IndexType idx = indexedFeature[iSample];
                auto& sum = aGHSum[idx];
                sum.n++;
                sum.g += pgh[iSample].g;
                sum.h += pgh[iSample].h;
                gTotal += pgh[iSample].g;
                hTotal += pgh[iSample].h;
            }
        }
    }
    //make a copy since it can be corrected below. TODO: propagate this corrected value to the argument?
    ImpurityType imp(job.imp);
    if(!isZero<algorithmFPType, cpu>(gTotal - imp.g))
        imp.g = gTotal;
    if(!isZero<algorithmFPType, cpu>(hTotal - imp.h))
        imp.h = hTotal;
    //index of best feature value in the array of sorted feature values
    int idxFeatureBestSplit = -1;
    //below we calculate only part of the impurity decrease dependent on split itself
    algorithmFPType bestImpDecrease = split.impurityDecrease;
    if(!split.featureUnordered)
    {
        size_t nLeft = 0;
        ImpurityType left;
        for(size_t i = 0; i < nDiffFeatMax; ++i)
        {
            if(!aGHSum[i].n)
                continue;
            nLeft += aGHSum[i].n;
            if((n - nLeft) < _ctx.par().minObservationsInLeafNode)
                break;
            left.add((const ghType&)aGHSum[i]);
            if(nLeft < _ctx.par().minObservationsInLeafNode)
                continue;

            ImpurityType right(imp, left);
            //the part of the impurity decrease dependent on split itself
            const algorithmFPType impDecrease = left.value(_ctx.par().lambda) + right.value(_ctx.par().lambda);
            if((impDecrease > bestImpDecrease) || (bUpdateWhenTie && (impDecrease == bestImpDecrease)))
            {
                split.left = left;
                split.nLeft = nLeft;
                idxFeatureBestSplit = i;
                bestImpDecrease = impDecrease;
            }
        }
        if(idxFeatureBestSplit >= 0)
            split.impurityDecrease = bestImpDecrease;
    }
    else
    {
        for(size_t i = 0; i < nDiffFeatMax; ++i)
        {
            if((aGHSum[i].n < _ctx.par().minObservationsInLeafNode) || ((n - aGHSum[i].n) < _ctx.par().minObservationsInLeafNode))
                continue;
            const ImpurityType& left = aGHSum[i];
            ImpurityType right(imp, left);
            //the part of the impurity decrease dependent on split itself
            const algorithmFPType impDecrease = left.value(_ctx.par().lambda) + right.value(_ctx.par().lambda);
            if(impDecrease > bestImpDecrease)
            {
                idxFeatureBestSplit = i;
                bestImpDecrease = impDecrease;
            }
        }
        if(idxFeatureBestSplit >= 0)
        {
            split.impurityDecrease = bestImpDecrease;
            split.nLeft = aGHSum[idxFeatureBestSplit].n;
            split.left = (const ghType&)aGHSum[idxFeatureBestSplit];
        }
    }
    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilder<algorithmFPType, cpu>::findBestSplitFeatSortedOrdered(const algorithmFPType* featureVal,
    const IndexType* aIdx, const SplitJobType& job, SplitDataType& split, bool bUpdateWhenTie) const
{
    ImpurityType left(_ctx.grad(this->_iTree)[*aIdx]);
    algorithmFPType bestImpurityDecrease = split.impurityDecrease;
    IndexType iBest = -1;
    const size_t n = job.n;
    const auto nMinSplitPart = _ctx.par().minObservationsInLeafNode;
    const algorithmFPType last = featureVal[n - nMinSplitPart];
    for(size_t i = 1; i < (n - nMinSplitPart + 1); ++i)
    {
        const bool bSameFeaturePrev(featureVal[i] <= featureVal[i - 1] + _ctx.accuracy());
        if(!(bSameFeaturePrev || i < nMinSplitPart))
        {
            //can make a split
            //nLeft == i, nRight == n - i
            ImpurityType right(job.imp, left);
            const algorithmFPType v = left.value(_ctx.par().lambda) + right.value(_ctx.par().lambda);
            if((v > bestImpurityDecrease) || (bUpdateWhenTie && (v == bestImpurityDecrease)))
            {
                bestImpurityDecrease = v;
                split.left = left;
                iBest = i;
            }
        }

        //update impurity and continue
        left.add(_ctx.grad(this->_iTree)[aIdx[i]]);
    }
    if(iBest < 0)
        return false;

    split.impurityDecrease = bestImpurityDecrease;
    split.nLeft = iBest;
    split.iStart = 0;
    split.featureValue = featureVal[iBest - 1];
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilder<algorithmFPType, cpu>::findBestSplitFeatSortedCategorical(const algorithmFPType* featureVal,
    const IndexType* aIdx, const SplitJobType& job, SplitDataType& split, bool bUpdateWhenTie) const
{
    const size_t n = job.n;
    const auto nMinSplitPart = _ctx.par().minObservationsInLeafNode;
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    algorithmFPType bestImpurityDecrease = split.impurityDecrease;
    ImpurityType left;
    bool bFound = false;
    size_t nDiffFeatureValues = 0;
    for(size_t i = 0; i < n - nMinSplitPart;)
    {
        ++nDiffFeatureValues;
        size_t count = 1;
        const algorithmFPType first = featureVal[i];
        const size_t iStart = i;
        for(++i; (i < n) && (featureVal[i] == first); ++count, ++i);
        if((count < nMinSplitPart) || ((n - count) < nMinSplitPart))
            continue;

        if((i == n) && (nDiffFeatureValues == 2) && bFound)
            break; //only 2 feature values, one possible split, already found

        calcImpurity(aIdx + iStart, count, left);
        ImpurityType right(job.imp, left);
        const algorithmFPType v = left.value(_ctx.par().lambda) + right.value(_ctx.par().lambda);
        if(v > bestImpurityDecrease || (bUpdateWhenTie && (v == bestImpurityDecrease)))
        {
            bestImpurityDecrease = v;
            split.left = left;
            split.nLeft = count;
            split.iStart = iStart;
            split.featureValue = first;
            bFound = true;
        }
    }
    if(bFound)
        split.impurityDecrease = bestImpurityDecrease;
    return bFound;
}

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
