/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <mutex>


#include "backend/cpu/CpuBackend.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/common/Tags.h"
#include "backend/cpu/Cpu.h"
#include "base/tools/Chrono.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxDataset.h"


#ifdef TNN_FEATURE_API
#   include "base/api/interfaces/IApiRequest.h"
#endif


#ifdef TNN_ALGO_ARGON2
#   include "crypto/argon2/Impl.h"
#endif


#ifdef TNN_FEATURE_BENCHMARK
#   include "backend/common/benchmark/Benchmark.h"
#   include "backend/common/benchmark/BenchState.h"
#endif


namespace xmrig {


extern template class Threads<CpuThreads>;


static const String kType   = "cpu";
static std::mutex mutex;


struct CpuLaunchStatus
{
public:
    inline const HugePagesInfo &hugePages() const   { return m_hugePages; }
    inline size_t memory() const                    { return m_ways * m_memory; }
    inline size_t threads() const                   { return m_threads; }
    inline size_t ways() const                      { return m_ways; }

    inline void start(const std::vector<CpuLaunchData> &threads, size_t memory)
    {
        m_workersMemory.clear();
        m_hugePages.reset();
        m_memory       = memory;
        m_started      = 0;
        m_totalStarted = 0;
        m_errors       = 0;
        m_threads      = threads.size();
        m_ways         = 0;
        m_ts           = Chrono::steadyMSecs();
    }

private:
    std::set<const VirtualMemory*> m_workersMemory;
    HugePagesInfo m_hugePages;
    size_t m_errors       = 0;
    size_t m_memory       = 0;
    size_t m_started      = 0;
    size_t m_totalStarted = 0;
    size_t m_threads      = 0;
    size_t m_ways         = 0;
    uint64_t m_ts         = 0;
};


class CpuBackendPrivate
{
public:
    inline explicit CpuBackendPrivate(Controller *controller) : controller(controller)   {}

    size_t ways() const
    {
        std::lock_guard<std::mutex> lock(mutex);

        return status.ways();
    }


    rapidjson::Value hugePages(int version, rapidjson::Document &doc) const
    {
        HugePagesInfo pages;

    #   ifdef TNN_ALGO_RANDOMX
        if (algo.family() == Algorithm::RANDOM_X) {
            pages += Rx::hugePages();
        }
    #   endif

        mutex.lock();

        pages += status.hugePages();

        mutex.unlock();

        rapidjson::Value hugepages;

        if (version > 1) {
            hugepages.SetArray();
            hugepages.PushBack(static_cast<uint64_t>(pages.allocated), doc.GetAllocator());
            hugepages.PushBack(static_cast<uint64_t>(pages.total), doc.GetAllocator());
        }
        else {
            hugepages = pages.isFullyAllocated();
        }

        return hugepages;
    }


    Algorithm algo;
    Controller *controller;
    CpuLaunchStatus status;
    std::vector<CpuLaunchData> threads;
    String profileName;

#   ifdef TNN_FEATURE_BENCHMARK
    std::shared_ptr<Benchmark> benchmark;
#   endif
};


} // namespace xmrig


xmrig::CpuBackend::~CpuBackend()
{
    delete d_ptr;
}

const xmrig::String &xmrig::CpuBackend::profileName() const
{
    return d_ptr->profileName;
}


const xmrig::String &xmrig::CpuBackend::type() const
{
    return kType;
}

void xmrig::CpuBackend::printHealth()
{
}

#ifdef TNN_FEATURE_API
rapidjson::Value xmrig::CpuBackend::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator         = doc.GetAllocator();
    const CpuConfig &cpu    = d_ptr->controller->config()->cpu();

    Value out(kObjectType);
    out.AddMember("type",       type().toJSON(), allocator);
    out.AddMember("enabled",    isEnabled(), allocator);
    out.AddMember("algo",       d_ptr->algo.toJSON(), allocator);
    out.AddMember("profile",    profileName().toJSON(), allocator);
    out.AddMember("hw-aes",     cpu.isHwAES(), allocator);
    out.AddMember("priority",   cpu.priority(), allocator);
    out.AddMember("msr",        Rx::isMSR(), allocator);

#   ifdef TNN_FEATURE_ASM
    const Assembly assembly = Cpu::assembly(cpu.assembly());
    out.AddMember("asm", assembly.toJSON(), allocator);
#   else
    out.AddMember("asm", false, allocator);
#   endif

#   ifdef TNN_ALGO_ARGON2
    out.AddMember("argon2-impl", argon2::Impl::name().toJSON(), allocator);
#   endif

    out.AddMember("hugepages", d_ptr->hugePages(2, doc), allocator);
    out.AddMember("memory",    static_cast<uint64_t>(d_ptr->algo.isValid() ? (d_ptr->ways() * d_ptr->algo.l3()) : 0), allocator);

    if (d_ptr->threads.empty() || !hashrate()) {
        return out;
    }

    out.AddMember("hashrate", hashrate()->toJSON(doc), allocator);

    Value threads(kArrayType);

    size_t i = 0;
    for (const CpuLaunchData &data : d_ptr->threads) {
        Value thread(kObjectType);
        thread.AddMember("intensity",   data.intensity, allocator);
        thread.AddMember("affinity",    data.affinity, allocator);
        thread.AddMember("av",          data.av(), allocator);
        thread.AddMember("hashrate",    hashrate()->toJSON(i, doc), allocator);

        i++;
        threads.PushBack(thread, allocator);
    }

    out.AddMember("threads", threads, allocator);

    return out;
}


void xmrig::CpuBackend::handleRequest(IApiRequest &request)
{
    if (request.type() == IApiRequest::REQ_SUMMARY) {
        request.reply().AddMember("hugepages", d_ptr->hugePages(request.version(), request.doc()), request.doc().GetAllocator());
    }
}
#endif


#ifdef TNN_FEATURE_BENCHMARK
xmrig::Benchmark *xmrig::CpuBackend::benchmark() const
{
    return d_ptr->benchmark.get();
}


void xmrig::CpuBackend::printBenchProgress() const
{
    if (d_ptr->benchmark) {
        d_ptr->benchmark->printProgress();
    }
}
#endif
