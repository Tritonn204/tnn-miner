#include <algo_definitions.h>
#include <stratum/pearl-stratum.hpp>
#include <tnn_hip/crypto/pearl/pearl_target.hpp>
#include <tnn_hip/common/gpu_job_mailbox.hpp>
#include <boost/json/src.hpp>
#include <cassert>
#include <iostream>
#include <thread>
#include <tnn_hip/coins/pearl/pearl_mining.hpp>

void test_atomic_jobs() {
    struct Snapshot {
        bool is_dev = false;
        uint64_t generation = 0;
        std::string id;
        std::array<uint8_t, 76> header{};
        std::array<uint8_t, 32> target{};
    };
    GPUJobMailbox<Snapshot> mailbox;
    std::atomic<unsigned> finished{0};
    auto writer = [&](bool dev) {
        for (uint64_t generation = 1; generation <= 20'000; ++generation) {
            Snapshot snapshot;
            snapshot.is_dev = dev;
            snapshot.generation = generation;
            snapshot.id = std::to_string(generation);
            snapshot.header.fill(uint8_t(generation));
            snapshot.target.fill(uint8_t(generation >> 8));
            mailbox.publish(std::move(snapshot));
        }
        ++finished;
    };
    std::thread user(writer, false);
    std::thread dev(writer, true);
    do {
        for (bool mode : {false, true}) {
            auto snapshot = mailbox.read(mode);
            if (!snapshot) continue;
            assert(snapshot->is_dev == mode);
            assert(snapshot->id == std::to_string(snapshot->generation));
            for (auto byte : snapshot->header) assert(byte == uint8_t(snapshot->generation));
            for (auto byte : snapshot->target) assert(byte == uint8_t(snapshot->generation >> 8));
        }
    } while (finished.load() != 2);
    user.join();
    dev.join();
    auto retained = mailbox.read(false);
    assert(retained && retained->generation == 20'000);
    mailbox.clear();
    assert(!mailbox.read(false) && !mailbox.read(true));
    assert(retained->generation == 20'000); // snapshot owns its lifetime
}

template <class Function>
void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

int main() {
    tnn::pearl::configure_mining();
    assert(tnn::pearl::mining_enabled);
    const tnn::pearl::ExecutionOptions defaults;
    assert(defaults.mode == tnn::pearl::ExecutionMode::Mining);
    assert(defaults.shape.m == 8192 && defaults.shape.n == 8192 && defaults.shape.k == 4096);
    assert(defaults.winner_capacity == 256);
    test_atomic_jobs();
    using namespace tnn::pearl::stratum;
    assert(completed_work(9) == 9);
    assert(completed_work(1, uint64_t(8192) * 8192 * 4096) == 274877906944ULL);
    assert(completed_work(0, UINT64_MAX) == 0);
    rejects([] { completed_work(2, UINT64_MAX); });
    rejects([] { completed_work(1, 0); });
    std::atomic<uint64_t> counter{UINT64_MAX - 1};
    add_completed_work(counter, 1);
    rejects([&] { add_completed_work(counter, 1); });
    assert(counter == UINT64_MAX);
    assert(algo_rate_info(ALGO_PEARL_POUW).unit == RateUnit::MultiplyAccumulates);
    assert(algo_rate_info(ALGO_KAWPOW).unit == RateUnit::Hashes);
    assert(std::string(rate_suffix(RateUnit::Solutions)) == "Sol/s");
    assert(std::string(efficiency_suffix(RateUnit::MultiplyAccumulates)) == "MAC/J");

    auto auth = authorize("test-wallet", "worker", "x", "tnn-test");
    assert(auth.at("params").is_object());
    boost::json::object notify{{"id", nullptr}, {"method", "mining.notify"},
        {"params", {{"job_id", "opaque"}, {"header", std::string(152, '0')},
                    {"target", std::string(62, '0') + "10"}, {"height", 1}, {"cert_version", 2}}}};
    auto job = parse_job(notify, 7);
    Jobs long_run(7);
    for (unsigned i = 0; i < 1000; ++i) {
        auto next = job;
        next.id = std::to_string(i);
        next.height = i;
        assert(long_run.update(next, Clock::now()));
        assert(long_run.eligible(next, Clock::now()));
        assert(!long_run.eligible(job, Clock::now()));
    }
    auto version_test = notify;
    auto& version_fields = version_test["params"].as_object();
    version_fields.erase("cert_version");
    rejects([&] { parse_job(version_test, 7); });
    assert(parse_job(version_test, 7, 2).cert_version == 2);
    for (auto bad : {boost::json::value(0), boost::json::value(4), boost::json::value(-1),
                     boost::json::value("3"), boost::json::value(3.0), boost::json::value(nullptr)}) {
        version_fields["cert_version"] = bad;
        rejects([&] { parse_job(version_test, 7, 2); });
    }
    version_fields["cert_version"] = 3;
    assert(parse_job(version_test, 7, 2).cert_version == 3);
    Jobs version_jobs(7);
    version_jobs.update(job, Clock::now());
    auto changed_version = job;
    changed_version.cert_version = 3;
    assert(version_jobs.update(changed_version, Clock::now()));
    assert(!version_jobs.eligible(job, Clock::now()));
    assert(job.target_le[0] == 16 && job.target_le[31] == 0);
    auto digest = job.target_le;
    assert(meets_target(digest, job.target_le));
    digest[0] = 17;
    assert(!meets_target(digest, job.target_le));
    digest[0] = 15;
    assert(meets_target(digest, job.target_le));
    digest[31] = 1;
    assert(!meets_target(digest, job.target_le));

    auto now = Clock::now();
    Jobs jobs(7);
    assert(jobs.update(job, now));
    assert(!jobs.update(job, now));
    auto next = job;
    next.id = "retarget";
    next.target_le[0] = 8;
    assert(jobs.update(next, now));
    assert(jobs.eligible(job, now + std::chrono::seconds(29)));
    assert(!jobs.eligible(job, now + std::chrono::seconds(30)));
    next.header[0] = 1;
    jobs.update(next, now);
    assert(!jobs.eligible(job, now));
    next.generation = 8;
    assert(!jobs.eligible(next, now));
    rejects([&] { jobs.update(next, now); });

    // Adjustment belongs to the owned attempt, never the mutable wire job.
    const auto original_wire = job.target_le;
    const auto work = tnn::pearl::jackpot_work(2, 64, 4096, 128);
    const auto in_flight_bound = tnn::pearl::scale_jackpot_target(original_wire, work);
    assert(job.target_le == original_wire);
    auto incorrectly_bound_job = job;
    incorrectly_bound_job.target_le = in_flight_bound;
    Jobs target_jobs(7);
    target_jobs.update(job, now);
    assert(!target_jobs.eligible(incorrectly_bound_job, now));
    auto retarget = job;
    retarget.id = "target-update";
    retarget.target_le[0] = 8;
    target_jobs.update(retarget, now);
    assert(target_jobs.eligible(job, now));
    assert(in_flight_bound == tnn::pearl::scale_jackpot_target(original_wire, work));
    assert(in_flight_bound != tnn::pearl::scale_jackpot_target(retarget.target_le, work));
    Jobs reconnected(8);
    assert(!reconnected.eligible(job, now));

    const auto wire = boost::json::serialize(notify) + "\n";
    Frames frames;
    assert(frames.feed(wire.substr(0, 9)).empty());
    assert(frames.feed(wire.substr(9) + wire).size() == 2);
    rejects([] { Frames{}.feed("[]\n"); });
    rejects([] { Frames{}.feed(std::string(max_frame_bytes, ' ')); });
    rejects([] { decode_hex<1>("gg"); });
    rejects([] { decode_hex<1>("0"); });
    notify["params"].as_object()["target"] = std::string(64, '0');
    rejects([&] { parse_job(notify, 7); });
    assert(submit(2, job, "AAAA").at("params").as_object().at("job_id") == "opaque");
    rejects([&] { submit(1, job, "AAAA"); });
    rejects([&] { submit(2, job, "A=AA"); });
    Connection connection(7);
    connection.jobs.update(job, now); // notify is legal before authorization ack
    rejects([&] { connection.begin_submit(2, job, 0, now); });
    auto reply = parse_reply(boost::json::parse(
        R"({"id":1,"result":true,"error":null})").as_object());
    assert(!connection.acknowledge(reply));
    assert(connection.authorized());
    connection.begin_submit(2, job, 3, now);
    rejects([&] { connection.begin_submit(3, job, 3, now); });
    reply = parse_reply(boost::json::parse(
        R"({"id":2,"result":null,"error":{"code":23,"msg":"low difficulty share"}})").as_object());
    assert(!reply.accepted && reply.error_code == 23);
    assert(connection.acknowledge(reply) == 3);
    rejects([&] { connection.acknowledge(reply); });
    rejects([&] { connection.begin_submit(2, job, 3, now); });
    rejects([] { parse_reply(boost::json::parse(R"({"id":3,"result":true})").as_object()); });
    std::cout << "PEARL_CONTRACT_PASS accounting, units, target, jobs, framing, submit, atomic_jobs\n";
}
