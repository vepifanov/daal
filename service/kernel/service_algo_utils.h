/* file: service_algo_utils.h */
/*******************************************************************************
* Copyright 2015-2018 Intel Corporation.
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
//  Declaration of service utilities used with services structures
//--
*/
#ifndef __SERVICE_ALGO_UTILS_H__
#define __SERVICE_ALGO_UTILS_H__

#include "services/host_app.h"
#include "error_indexes.h"
#include "algorithms/algorithm_types.h"

namespace daal
{
namespace services
{
namespace internal
{

services::HostAppIface* hostApp(algorithms::Input& inp);
void setHostApp(const services::SharedPtr<services::HostAppIface>& pHostApp, algorithms::Input& inp);
services::HostAppIfacePtr getHostApp(daal::algorithms::Input& inp);

inline bool isCancelled(services::Status& s, services::HostAppIface* pHostApp)
{
    if(!pHostApp || !pHostApp->isCancelled())
        return false;
    s.add(services::ErrorUserCancelled);
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Helper class handling cancellation status depending on the number of jobs to be done
//////////////////////////////////////////////////////////////////////////////////////////
class HostAppHelper
{
public:
    HostAppHelper(HostAppIface* hostApp, size_t maxJobsBeforeCheck) :
        _hostApp(hostApp), _maxJobsBeforeCheck(maxJobsBeforeCheck),
        _nJobsAfterLastCheck(0)
    {
    }
    bool isCancelled(services::Status& s, size_t nJobsToDo)
    {
        if(!_hostApp)
            return false;
        _nJobsAfterLastCheck += nJobsToDo;
        if(_nJobsAfterLastCheck < _maxJobsBeforeCheck)
            return false;
        _nJobsAfterLastCheck = 0;
        return services::internal::isCancelled(s, _hostApp);
    }

    void setup(size_t maxJobsBeforeCheck)
    {
        _maxJobsBeforeCheck = maxJobsBeforeCheck;
    }

    void reset(size_t maxJobsBeforeCheck)
    {
        setup(maxJobsBeforeCheck);
        _nJobsAfterLastCheck = 0;
    }

private:
    services::HostAppIface* _hostApp;
    size_t _maxJobsBeforeCheck; //granularity
    size_t _nJobsAfterLastCheck;
};

} // namespace internal
} // namespace services
} // namespace daal

#endif
