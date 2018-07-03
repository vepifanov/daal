/* file: service_threading.h */
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
//  Declaration of service threding classes and utilities
//--
*/
#ifndef __SERVICE_THREADING_H__
#define __SERVICE_THREADING_H__
#include "threading.h"

namespace daal
{

class Mutex
{
public:
    Mutex();
    ~Mutex();
    void lock();
    void unlock();
private:
    void* _impl;
};

class AutoLock
{
public:
    AutoLock(Mutex& m) : _m(m){ _m.lock(); }
    ~AutoLock() { _m.unlock(); }
private:
    Mutex& _m;
};

#define AUTOLOCK(m) AutoLock __autolock(m);

template<typename F>
class task_impl : public task
{
public:
    DAAL_NEW_DELETE();
    virtual void run()
    {
        _func();
    }
    virtual void destroy()
    {
        delete this;
    }
    static task_impl<F>* create(const F& o)
    {
        return new task_impl<F>(o);
    }

private:
    task_impl(const F& o) : task(), _func(o){}
    F _func;
};

class task_group
{
public:
    task_group() : _impl(NULL)
    {
        _impl = _daal_new_task_group();
    }
    ~task_group()
    {
        if(_impl)
            _daal_del_task_group(_impl);
    }
    template<typename F>
    void run(F &f)
    {
        if(_impl)
            _daal_run_task_group(_impl, task_impl<F>::create(f));
        else
            f();
    }
    void wait()
    {
        if(_impl)
            _daal_wait_task_group(_impl);
    }

protected:
    void* _impl;
};

} // namespace daal

#endif
