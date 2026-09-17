#include "../net.hpp"
#include <tnn_log.hpp>
#include <stratum/pearl-stratum.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/asio/ssl/host_name_verification.hpp>
#include <boost/beast/ssl.hpp>
#include <array>
#include <memory>
#include <tnn_hip/coins/pearl/pearl_mining.hpp>
#include <tnn_hip/coins/pearl/pearl_logging.hpp>
#include <tnn_hip/coins/pearl/pearl_share_audit.hpp>
#ifdef _WIN32
#include <wincrypt.h>
#endif

namespace tnn::pearl {
namespace {

std::atomic<uint64_t> next_generation{1};

void load_system_roots(ssl::context& context) {
#ifdef _WIN32
    // The legacy TNN helper embeds only one CA. Use the operating system's
    // trusted roots for this verified public TLS connection; never verify_none.
    HCERTSTORE roots = CertOpenSystemStoreA(0, "ROOT");
    if (!roots) throw std::runtime_error("Cannot open Windows trusted roots");
    PCCERT_CONTEXT certificate = nullptr;
    auto* store = SSL_CTX_get_cert_store(context.native_handle());
    while ((certificate = CertEnumCertificatesInStore(roots, certificate)) != nullptr) {
        const unsigned char* bytes = certificate->pbCertEncoded;
        X509* parsed = d2i_X509(nullptr, &bytes, certificate->cbCertEncoded);
        if (parsed) {
            (void)X509_STORE_add_cert(store, parsed);
            X509_free(parsed);
        }
    }
    CertCloseStore(roots, 0);
    ERR_clear_error(); // Duplicate trusted roots are harmless.
#else
    context.set_default_verify_paths();
#endif
}

// Like the Xelis/KawPow sessions, use TNN's job globals and submit handoff.
// Unlike a thread holding a borrowed stream, all I/O handlers own the session
// and run on one strand. Only one write can be outstanding.
template <class Stream>
class Session : public std::enable_shared_from_this<Session<Stream>> {
public:
    Session(net::io_context& ioc, std::unique_ptr<Stream> stream, bool is_dev,
            uint64_t generation)
        : stream_(std::move(stream)), strand_(net::make_strand(ioc)), timer_(ioc),
          state_(generation), generation_(generation), is_dev_(is_dev) {}

    void start(boost::json::object auth) {
        net::post(strand_, [self = this->shared_from_this(), auth = std::move(auth)] {
            self->write(auth);
            self->read();
            self->tick();
        });
    }

    void stop() {
        net::post(strand_, [self = this->shared_from_this()] { self->finish(); });
    }

    bool stopped() const { return stopped_.load(); }
    bool invalid_wallet() const { return invalid_wallet_.load(); }

private:
    void finish() {
        if (stopped_.exchange(true)) return;
        if (state_.submission_pending()) ++share_audit.ambiguous;
        boost::system::error_code ignored;
        timer_.cancel(ignored);
        beast::get_lowest_layer(*stream_).socket().cancel(ignored);
        beast::get_lowest_layer(*stream_).socket().close(ignored);
        std::scoped_lock guard(mutex);
        (is_dev_ ? devConnected : isConnected) = false;
        if (is_dev_ ? submittingDev : submitting) ++share_audit.cancelled;
        (is_dev_ ? submittingDev : submitting) = false;
        // The session's request map is discarded. Ambiguous submits are not replayed.
    }

    void failure(const std::string& message) {
        log_stratum_error(message);
        finish();
    }

    void publish() {
        if (!state_.authorized() || !latest_params_) return;
        auto fields = *latest_params_;
        fields["connection_generation"] = generation_;
        const double share_diff = stratum::share_difficulty(state_.jobs.current()->target_le);
        std::scoped_lock guard(mutex);
        (is_dev_ ? devJob : job) = fields;
        (is_dev_ ? doubleDiffDev : doubleDiff) = share_diff;
        (is_dev_ ? devConnected : isConnected) = true;
        if (is_dev_) ++devHeight;
        else { ++ourHeight; ++jobCounter; }
    }

    void receive(const boost::json::object& packet) {
        if (const auto* method = packet.if_contains("method")) {
            stratum::require(method->is_string(), "Invalid notification method");
            if (method->as_string() != "mining.notify") return;
            auto next = stratum::parse_job(packet, generation_, cert_version_fallback);
            const auto version = next.cert_version;
            if (state_.jobs.update(std::move(next), stratum::Clock::now())) {
                latest_params_ = packet.at("params").as_object();
                const bool fallback_used = !latest_params_->contains("cert_version");
                TNN_LOG_TRACE("[PEARL-JOB] generation=%llu dev=%d effective_cert_version=%u derivation=%s fallback=%d packet=%s\n",
                    static_cast<unsigned long long>(generation_), int(is_dev_), version,
                    version == 3 ? "salted" : "legacy", int(fallback_used), boost::json::serialize(packet).c_str());
                (*latest_params_)["cert_version"] = version;
                publish();
            }
            return;
        }
        TNN_LOG_TRACE("[PEARL-ACK] dev=%d packet=%s\n", int(is_dev_), boost::json::serialize(packet).c_str());
        TNN_LOG_TRACE("[PEARL-ACK-BIND] generation=%llu dev=%d packet=%s\n",
            static_cast<unsigned long long>(generation_), int(is_dev_), boost::json::serialize(packet).c_str());
        const auto reply = stratum::parse_reply(packet);
        const auto device = state_.acknowledge(reply);
        if (reply.id == 1) {
            if (!reply.accepted) {
                invalid_wallet_ = reply.error_code == 24 || reply.error_code == 25;
                failure("Authorization rejected: " + reply.error_message);
            } else {
                publish(); // Also handles notify-before-authorize-ack.
            }
        } else if (device) {
            if (reply.accepted) ++share_audit.accepted;
            else ++share_audit.rejected;
            if (!is_dev_) {
                std::scoped_lock guard(mutex);
                if (reply.accepted) ++accepted;
                else ++rejected;
                recordDeviceShare(*device, reply.accepted);
            }
            log_share(is_dev_, *device, reply.accepted, reply.error_message);
        }
    }

    void read() {
        if (stopped()) return;
        stream_->async_read_some(net::buffer(input_), net::bind_executor(strand_,
            [self = this->shared_from_this()](boost::system::error_code error, size_t size) {
                if (self->stopped()) return;
                if (error) { self->failure(error.message()); return; }
                try {
                    for (const auto& packet : self->frames_.feed({self->input_.data(), size})) {
                        self->receive(packet);
                        if (self->stopped()) return;
                    }
                    self->read();
                } catch (const std::exception& e) { self->failure(e.what()); }
            }));
    }

    void write(const boost::json::object& packet) {
        stratum::require(!writing_, "Overlapping Stratum writes");
        output_ = boost::json::serialize(packet) + "\n";
        stratum::require(output_.size() <= stratum::max_frame_bytes, "Oversized submit frame");
        writing_ = true;
        request_started_ = stratum::Clock::now();
        net::async_write(*stream_, net::buffer(output_), net::bind_executor(strand_,
            [self = this->shared_from_this()](boost::system::error_code error, size_t) {
                self->writing_ = false;
                if (!self->stopped() && error) self->failure(error.message());
            }));
    }

    void poll_submit() {
        if (writing_ || !state_.authorized() || state_.submission_pending()) return;
        boost::json::object candidate;
        {
            std::scoped_lock guard(mutex);
            bool& pending = is_dev_ ? submittingDev : submitting;
            if (!data_ready || !pending) return;
            candidate = is_dev_ ? devShare : share;
            pending = false;
            data_ready = submitting || submittingDev;
        }
        // Private metadata is required for immutable job binding and is never sent
        // to the pool. The future proof builder supplies this alongside params.
        struct CandidateGuard {
            bool settled = false;
            ~CandidateGuard() { if (!settled) ++share_audit.failed; }
        } audit_guard;
        const auto& binding = candidate.at("_pearl_job").as_object();
        boost::json::object notification{{"method", "mining.notify"}, {"params", binding}};
        const auto generation = binding.at("connection_generation").to_number<uint64_t>();
        const auto original = stratum::parse_job(notification, generation);
        if (!state_.jobs.eligible(original, stratum::Clock::now())) {
            ++share_audit.stale;
            audit_guard.settled = true;
            return;
        }
        const int device = candidate.at("_pearl_device").to_number<int>();
        const auto& params = candidate.at("params").as_object();
        stratum::require(stratum::string_field(params, "job_id") == original.id,
                         "Submission job identity mismatch");
        const auto id = ++request_id_;
        if (tnn_log_enabled(TnnLogLevel::Trace) &&
            candidate.contains("_pearl_evidence") && candidate.contains("_pearl_digest")) {
            boost::json::object receipt{{"id", id}, {"generation", generation_}, {"dev", is_dev_},
                {"job", original.id}, {"evidence", candidate.at("_pearl_evidence")}, {"digest", candidate.at("_pearl_digest")}};
            TNN_LOG_TRACE("[PEARL-SUBMIT-BIND] %s\n", boost::json::serialize(receipt).c_str());
        }
        auto packet = stratum::submit(id, original, stratum::string_field(params, "plain_proof"));
        state_.begin_submit(id, original, device, stratum::Clock::now());
        ++share_audit.submitted;
        audit_guard.settled = true;
        TNN_LOG_DEBUG("[PEARL-SUBMIT] dev=%d id=%llu job=%s\n", int(is_dev_),
                     static_cast<unsigned long long>(id), original.id.c_str());
        write(packet);
    }

    void tick() {
        if (stopped()) return;
        timer_.expires_after(std::chrono::milliseconds(25));
        timer_.async_wait(net::bind_executor(strand_,
            [self = this->shared_from_this()](boost::system::error_code error) {
                if (error || self->stopped()) return;
                try {
                    if (ABORT_MINER) { self->finish(); return; }
                    const bool awaiting = !self->state_.authorized() || self->writing_ ||
                                          self->state_.submission_pending();
                    if (awaiting && stratum::Clock::now() - self->request_started_ >
                                    std::chrono::seconds(60)) {
                        self->failure("Request timed out; not replaying submission");
                        return;
                    }
                    self->poll_submit();
                    self->tick();
                } catch (const std::exception& e) { self->failure(e.what()); }
            }));
    }

    std::unique_ptr<Stream> stream_;
    net::strand<net::io_context::executor_type> strand_;
    net::steady_timer timer_;
    stratum::Connection state_;
    stratum::Frames frames_;
    uint64_t generation_;
    bool is_dev_;
    bool writing_ = false;
    uint64_t request_id_ = 1;
    std::atomic<bool> stopped_{false};
    std::atomic<bool> invalid_wallet_{false};
    std::optional<boost::json::object> latest_params_;
    std::array<char, 8192> input_{};
    std::string output_;
    stratum::Clock::time_point request_started_ = stratum::Clock::now();
};

template <class Stream>
void run_connected(net::io_context& ioc, net::yield_context yield,
                   std::unique_ptr<Stream> stream, const std::string& wallet,
                   const std::string& worker, bool is_dev) {
    auto session = std::make_shared<Session<Stream>>(ioc, std::move(stream), is_dev,
                                                    next_generation.fetch_add(1));
    session->start(stratum::authorize(wallet, worker, stratum::authorization_password(is_dev, stratumPassword),
                                      "tnn-miner/" + std::string(versionString)));
    net::steady_timer wait(ioc);
    while (!session->stopped()) {
        if (ABORT_MINER) session->stop();
        wait.expires_after(std::chrono::milliseconds(50));
        boost::system::error_code error;
        wait.async_wait(yield[error]);
        if (error) { session->stop(); break; }
    }
    // Invalid credentials should not trigger the outer reconnect loop.
    if (session->invalid_wallet() && !is_dev) ABORT_MINER = true;
}

} // namespace

void pearl_stratum_session(std::string host, const std::string& port,
                            const std::string& wallet, const std::string& worker,
                            net::io_context& ioc, ssl::context& context,
                            net::yield_context yield, bool is_dev, bool use_ssl) {
    const auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
    boost::system::error_code error;
    if (use_ssl) {
        load_system_roots(context);
        auto stream = std::make_unique<beast::ssl_stream<beast::tcp_stream>>(ioc, context);
        stream->set_verify_mode(ssl::verify_peer);
        stream->set_verify_callback(ssl::host_name_verification(host));
        if (!SSL_set_tlsext_host_name(stream->native_handle(), host.c_str())) {
            throw std::runtime_error("Pearl TLS SNI setup failed");
        }
        beast::get_lowest_layer(*stream).expires_after(std::chrono::seconds(30));
        beast::get_lowest_layer(*stream).async_connect(endpoint, yield[error]);
        if (error) return fail(error, "Pearl TLS connect");
        stream->async_handshake(ssl::stream_base::client, yield[error]);
        if (error) return fail(error, "Pearl TLS handshake");
        beast::get_lowest_layer(*stream).expires_never();
        run_connected(ioc, yield, std::move(stream), wallet, worker, is_dev);
    } else {
        auto stream = std::make_unique<beast::tcp_stream>(ioc);
        stream->expires_after(std::chrono::seconds(30));
        stream->async_connect(endpoint, yield[error]);
        if (error) return fail(error, "Pearl connect");
        stream->expires_never();
        run_connected(ioc, yield, std::move(stream), wallet, worker, is_dev);
    }
}

} // namespace tnn::pearl
