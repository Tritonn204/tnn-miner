#pragma once
// Thread-safe access to the global job/devJob state.
//
// The globals `job`, `devJob`, `ourHeight`, `devHeight`, `difficulty`,
// `isConnected`, `devConnected` etc. are written by network threads and
// read by mining threads.  All access must go through the global `mutex`
// declared in net/net.hpp.
//
// These macros standardise the locking pattern so every call site is
// consistent and grep-able.

#include <mutex>
#include <boost/json/value.hpp>

// ---- Mining-side: snapshot globals into local copies --------------------

// Copy both job values under the lock.
#define TNN_SNAPSHOT_JOBS(myJob, myJobDev) \
    do { \
        std::scoped_lock<std::mutex> _job_guard(mutex); \
        (myJob)   = job; \
        (myJobDev) = devJob; \
    } while (0)

// ---- Network-side: read a job global into a local object ---------------

// Read (*JV).as_object() under the lock — use when building a modified
// copy of the current job before writing it back.
#define TNN_READ_JOB_OBJ(J, JV) \
    do { \
        std::scoped_lock<std::mutex> _job_guard(mutex); \
        (J) = (*(JV)).as_object(); \
    } while (0)

// ---- Network-side: write back a modified job object --------------------
// (Most call sites already lock before the write — keep using scoped_lock
//  directly for those since they often update height/counter in the same
//  critical section.)
