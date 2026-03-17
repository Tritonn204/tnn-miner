#pragma once
#include <tnn_hip/common/gpu_compat.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <algorithm>
#include <functional>
#include <optional>
#include "tnn_log.hpp"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif

class RTCCompiler {
public:
    struct CompiledKernel {
        oroModule_t module = nullptr;
        oroFunction_t function = nullptr;
        std::string kernel_name;
        bool from_cache = false;
    };

    struct CompiledCode {
        std::vector<char> code;
        std::string kernel_name;
    };

    static RTCCompiler& instance() {
        static RTCCompiler inst;
        return inst;
    }

    // ---------------------------------------------------------------------
    // Header registration helpers
    // ---------------------------------------------------------------------

    void add_header_file(const std::string& header_path) {
        add_header_file_internal("", header_path);
    }

    void add_header_file(const std::string& include_name,
                         const std::string& header_path) {
        add_header_file_internal(include_name, header_path);
    }

    void add_header_source(const std::string& include_name,
                           const std::string& header_source) {
        if (include_name.empty()) {
            throw std::runtime_error("include_name cannot be empty for add_header_source");
        }

        // Idempotent: if already registered, do nothing.
        for (const auto& h : headers_) {
            if (h.name == include_name) {
                return;
            }
        }

        Header h;
        h.name   = include_name;
        h.source = header_source;
        headers_.push_back(std::move(h));
    }

    // ---------------------------------------------------------------------
    // Compile
    // ---------------------------------------------------------------------

    // Compile from embedded source (preferred for HIPRTC)
    CompiledKernel compile_from_source(
        const std::string& source,
        const std::string& source_name,
        const std::string& kernel_name,
        const std::vector<std::string>& extra_options = {},
        int device_id = 0
    ) {
        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_from_source: Entry\n");
        TNN_LOG_TRACE("[TRACE]   source_name: %s\n", source_name.c_str());
        TNN_LOG_TRACE("[TRACE]   kernel_name: %s\n", kernel_name.c_str());
        TNN_LOG_TRACE("[TRACE]   source size: %zu\n", source.size());
        fflush(stdout);

        // Build normalized options
        const auto defaults = build_default_options(device_id);
        const auto norm = normalize_options(defaults, extra_options);

        // Module cache key (includes device-specific options like -DDEVICE_ID)
        const std::string module_key = make_cache_key(source_name, kernel_name, norm.sorted);

        // Code cache key (excludes device-specific options)
        auto code_options = filter_device_specific_options(norm.sorted);
        const std::string code_key = make_cache_key(source_name, kernel_name, code_options);

        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_from_source: Checking caches\n");
        TNN_LOG_TRACE("[TRACE]   module_key='%s'\n", module_key.c_str());
        TNN_LOG_TRACE("[TRACE]   code_key='%s'\n", code_key.c_str());
        fflush(stdout);

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // Always reload module from code cache — oroModuleLoadData binds
            // the module to the calling thread's GPU context.
            auto code_it = code_cache_.find(code_key);
            if (code_it != code_cache_.end()) {
                TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_from_source: Found in code cache, loading module\n");
                fflush(stdout);

                CompiledKernel kernel = load_module_from_code(code_it->second);
                kernel.from_cache = true;
                module_cache_[module_key] = kernel;
                return kernel;
            }
        }

        // AMD disk cache: check before expensive compilation
        if (tnn_is_amd_device(device_id)) {
            std::string disk_hash = make_disk_cache_hash(source, kernel_name, code_options);
            auto disk_code = load_from_disk_cache(disk_hash, kernel_name);
            if (disk_code.has_value()) {
                TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[PRECOMPILE] Disk cache hit: %s\n", kernel_name.c_str());
                fflush(stdout);
                try {
                    CompiledKernel kernel = load_module_from_code(disk_code.value());
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    code_cache_[code_key] = std::move(disk_code.value());
                    module_cache_[module_key] = kernel;
                    return kernel;
                } catch (...) {
                    TNN_LOG_DEBUG("[PRECOMPILE] Disk cache binary failed to load, recompiling\n");
                    // Fall through to compile
                }
            }
        }

        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_from_source: Not in cache, calling compile_internal\n");
        fflush(stdout);

        return compile_internal(source, source_name, kernel_name, extra_options, module_key, code_key, device_id);
    }

    // Compile from file path (fallback)
    CompiledKernel compile(
        const std::string& source_path,
        const std::string& kernel_name,
        const std::vector<std::string>& extra_options = {},
        int device_id = 0
    ) {
        // Build normalized options
        const auto defaults = build_default_options(device_id);
        const auto norm = normalize_options(defaults, extra_options);

        // Module cache key (includes device-specific options)
        const std::string module_key = make_cache_key(source_path, kernel_name, norm.sorted);

        // Code cache key (excludes device-specific options)
        auto code_options = filter_device_specific_options(norm.sorted);
        const std::string code_key = make_cache_key(source_path, kernel_name, code_options);

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);

            auto code_it = code_cache_.find(code_key);
            if (code_it != code_cache_.end()) {
                CompiledKernel kernel = load_module_from_code(code_it->second);
                module_cache_[module_key] = kernel;
                return kernel;
            }
        }

        std::string source = load_text_file(source_path);
        return compile_internal(source, source_path, kernel_name, extra_options, module_key, code_key, device_id);
    }

    void clear_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto& kv : module_cache_) {
            if (kv.second.module) {
                (void)oroModuleUnload(kv.second.module);
            }
        }
        module_cache_.clear();
        code_cache_.clear();
    }

private:
    struct Header {
        std::string name;   // include name, e.g. "xelis_common.h"
        std::string source; // full header text
    };

    // ----------------------------
    // Option normalization helpers
    // ----------------------------

    static inline bool starts_with(const std::string& s, const char* pfx) {
        return s.rfind(pfx, 0) == 0;
    }

    // Return a "dedupe key" for keyed options, else empty for plain options.
    // Keyed options are last-one-wins.
    static std::string option_key(const std::string& opt) {
        if (starts_with(opt, "--gpu-architecture=")) return "--gpu-architecture";
        if (starts_with(opt, "--maxrregcount="))     return "--maxrregcount";
        if (starts_with(opt, "-std="))               return "-std";
        if (starts_with(opt, "--std="))              return "--std";
        if (starts_with(opt, "-O"))                  return "-O";

        // Macro defines: last-one-wins per macro name
        // -DFOO / -DFOO=1 / -DFOO=bar  -> key "-D:FOO"
        if (starts_with(opt, "-D")) {
            std::string rest = opt.substr(2);
            size_t eq = rest.find('=');
            std::string name = (eq == std::string::npos) ? rest : rest.substr(0, eq);
            if (!name.empty()) return std::string("-D:") + name;
        }

        return {};
    }

    struct NormalizedOptions {
        std::vector<std::string> ordered; // deterministic compile order
        std::vector<std::string> sorted;  // stable sorted representation for cache key
    };

    static NormalizedOptions normalize_options(
        const std::vector<std::string>& defaults,
        const std::vector<std::string>& extras
    ) {
        std::unordered_map<std::string, std::string> keyed; // last-one-wins
        std::unordered_set<std::string> plain_set;          // exact dedupe

        auto ingest = [&](const std::vector<std::string>& src) {
            for (const auto& o : src) {
                if (o.empty()) continue;
                std::string k = option_key(o);
                if (!k.empty()) {
                    keyed[k] = o;
                } else {
                    plain_set.insert(o);
                }
            }
        };

        // extras override defaults
        ingest(defaults);
        ingest(extras);

        std::vector<std::string> out;
        out.reserve(keyed.size() + plain_set.size());

        auto emit_key = [&](const char* k) {
            auto it = keyed.find(k);
            if (it != keyed.end()) {
                out.push_back(it->second);
                keyed.erase(it);
            }
        };

        // Preferred deterministic ordering for common knobs
        emit_key("-std");
        emit_key("--std");
        emit_key("-O");
        emit_key("--gpu-architecture");
        emit_key("--maxrregcount");

        // Remaining keyed options sorted by key name
        {
            std::vector<std::pair<std::string, std::string>> rest;
            rest.reserve(keyed.size());
            for (auto& kv : keyed) rest.push_back(kv);

            std::sort(rest.begin(), rest.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            for (auto& kv : rest) out.push_back(kv.second);
        }

        // Plain options sorted by option string
        {
            std::vector<std::string> plain(plain_set.begin(), plain_set.end());
            std::sort(plain.begin(), plain.end());
            for (auto& p : plain) out.push_back(p);
        }

        // Cache-key form: fully sorted by string so cache never depends on caller ordering
        std::vector<std::string> sorted_key = out;
        std::sort(sorted_key.begin(), sorted_key.end());

        return { out, sorted_key };
    }

    // ----------------------------
    // Construction / helpers
    // ----------------------------

    RTCCompiler() {
        init_disk_cache_dir();
    }

    ~RTCCompiler() {
        clear_cache();
    }

    static std::string load_text_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        std::ostringstream oss;
        oss << file.rdbuf();
        return oss.str();
    }

    void add_header_file_internal(const std::string& explicit_name,
                                  const std::string& header_path) {
        Header h;
        if (!explicit_name.empty()) {
            h.name = explicit_name;
        } else {
            h.name = std::filesystem::path(header_path).filename().string();
        }
        h.source = load_text_file(header_path);
        headers_.push_back(std::move(h));
    }

    std::vector<std::string> build_default_options(int device_id = 0) const {
        std::vector<std::string> d;

        bool is_amd = tnn_is_amd_device(device_id);

        if (is_amd) {
            d.emplace_back("-O3");
        }
        d.emplace_back("-std=c++20");

        if (is_amd) {
            d.emplace_back("-DHIP_PLATFORM_AMD");
        } else {
            d.emplace_back("-DHIP_PLATFORM_NVIDIA");
        }

        return d;
    }

    static std::string make_cache_key(
        const std::string& source_id,
        const std::string& kernel_name,
        const std::vector<std::string>& normalized_sorted_options
    ) {
        std::string key = source_id + ":" + kernel_name;
        for (const auto& opt : normalized_sorted_options) {
            key.push_back('+');
            key.append(opt);
        }
        return key;
    }

    // Filter out device-specific options for code cache (allows sharing across identical GPUs)
    static std::vector<std::string> filter_device_specific_options(
        const std::vector<std::string>& options
    ) {
        std::vector<std::string> filtered;
        filtered.reserve(options.size());
        for (const auto& opt : options) {
            // Skip device-specific options that don't affect compilation output
            if (starts_with(opt, "-DDEVICE_ID=")) continue;
            filtered.push_back(opt);
        }
        return filtered;
    }

    // Load module from cached compiled code
    CompiledKernel load_module_from_code(const CompiledCode& cached_code) {
        TNN_LOG_TRACE("[TRACE] RTCCompiler::load_module_from_code: Loading module from %zu bytes\n",
               cached_code.code.size());
        fflush(stdout);

        CompiledKernel kernel;
        kernel.kernel_name = cached_code.kernel_name;

        oroError_t err = oroModuleLoadData(&kernel.module, cached_code.code.data());
        if (err != oroSuccess) {
            printf("[ERROR] oroModuleLoadData failed: %d (%s)\n", err, tnn_error_string(err));
            fflush(stdout);
            throw std::runtime_error("oroModuleLoadData failed: " + std::string(tnn_error_string(err)));
        }

        err = oroModuleGetFunction(&kernel.function, kernel.module, kernel.kernel_name.c_str());
        if (err != oroSuccess) {
            (void)oroModuleUnload(kernel.module);
            throw std::runtime_error("Failed to get kernel function: " + kernel.kernel_name);
        }

        TNN_LOG_TRACE("[TRACE] RTCCompiler::load_module_from_code: Module loaded successfully\n");
        fflush(stdout);

        return kernel;
    }

    // ----------------------------
    // Disk cache (AMD only)
    // ----------------------------

    // FNV-1a 64-bit hash — simple, fast, good distribution for cache keys
    static uint64_t fnv1a_hash(const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        uint64_t h = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < len; ++i) {
            h ^= p[i];
            h *= 0x100000001b3ULL;
        }
        return h;
    }

    static std::string to_hex16(uint64_t v) {
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
        return buf;
    }

    // Hash source + all headers + options into a stable hex string for filename
    std::string make_disk_cache_hash(
        const std::string& source,
        const std::string& kernel_name,
        const std::vector<std::string>& sorted_options
    ) const {
        // Build a single blob to hash: source + headers (sorted by name) + options + kernel name
        std::string blob;
        blob.reserve(source.size() + 4096);
        blob.append(source);
        blob.push_back('\0');
        blob.append(kernel_name);
        blob.push_back('\0');

        // Sort headers by name for deterministic ordering
        std::vector<const Header*> sorted_headers;
        sorted_headers.reserve(headers_.size());
        for (const auto& h : headers_) sorted_headers.push_back(&h);
        std::sort(sorted_headers.begin(), sorted_headers.end(),
                  [](const Header* a, const Header* b) { return a->name < b->name; });

        for (const auto* h : sorted_headers) {
            blob.append(h->name);
            blob.push_back('\0');
            blob.append(h->source);
            blob.push_back('\0');
        }

        for (const auto& opt : sorted_options) {
            blob.append(opt);
            blob.push_back('\0');
        }

        // Use two independent FNV-1a passes (different seeds via offset) for 128-bit key
        uint64_t h1 = fnv1a_hash(blob.data(), blob.size());
        // Second hash: offset by 1 byte to get independent hash
        uint64_t h2 = fnv1a_hash(blob.data() + 1, blob.size() - 1);

        return to_hex16(h1) + to_hex16(h2);
    }

    void init_disk_cache_dir() {
        // Check env var first (set by wrapper scripts on HiveOS/mmpOS)
        const char* env_path = std::getenv("TNN_HIP_CACHE_PATH");
        if (env_path && env_path[0]) {
            disk_cache_dir_ = env_path;
        } else {
#ifdef _WIN32
            // %LOCALAPPDATA%/TNN-Miner/HipCache
            char appdata[MAX_PATH] = {0};
            if (SHGetFolderPathA(nullptr, CSIDL_LOCAL_APPDATA, nullptr, 0, appdata) == S_OK) {
                disk_cache_dir_ = std::string(appdata) + "\\TNN-Miner\\HipCache";
            } else {
                const char* tmp = std::getenv("TEMP");
                disk_cache_dir_ = std::string(tmp ? tmp : ".") + "\\TNN-Miner\\HipCache";
            }
#else
            // ~/.cache/tnn-miner/hip_cache
            const char* home = std::getenv("HOME");
            if (home) {
                disk_cache_dir_ = std::string(home) + "/.cache/tnn-miner/hip_cache";
            } else {
                disk_cache_dir_ = "/tmp/tnn-miner/hip_cache";
            }
#endif
        }

        // Create directory if it doesn't exist
        std::error_code ec;
        std::filesystem::create_directories(disk_cache_dir_, ec);
        if (ec) {
            TNN_LOG_DEBUG("[DISK CACHE] Failed to create cache dir '%s': %s\n",
                          disk_cache_dir_.c_str(), ec.message().c_str());
            disk_cache_dir_.clear(); // Disable disk cache
        } else {
            TNN_LOG_INFO("[DISK CACHE] Using: %s\n", disk_cache_dir_.c_str());
        }
    }

    std::optional<CompiledCode> load_from_disk_cache(
        const std::string& hash,
        const std::string& kernel_name
    ) {
        if (disk_cache_dir_.empty()) return std::nullopt;

        std::string path = disk_cache_dir_ +
#ifdef _WIN32
            "\\"
#else
            "/"
#endif
            + hash + ".bin";

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return std::nullopt;

        auto size = file.tellg();
        if (size <= 0 || size > 256 * 1024 * 1024) return std::nullopt; // sanity: max 256MB

        file.seekg(0);
        CompiledCode code;
        code.code.resize(static_cast<size_t>(size));
        code.kernel_name = kernel_name;

        if (!file.read(code.code.data(), size)) return std::nullopt;

        TNN_LOG_DEBUG("[DISK CACHE] Loaded %zu bytes from %s\n",
                      code.code.size(), path.c_str());
        return code;
    }

    void save_to_disk_cache(const std::string& hash, const std::vector<char>& code) {
        if (disk_cache_dir_.empty()) return;

        std::string path = disk_cache_dir_ +
#ifdef _WIN32
            "\\"
#else
            "/"
#endif
            + hash + ".bin";

        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            TNN_LOG_DEBUG("[DISK CACHE] Failed to write %s\n", path.c_str());
            return;
        }

        file.write(code.data(), code.size());
        TNN_LOG_DEBUG("[DISK CACHE] Saved %zu bytes to %s\n", code.size(), path.c_str());
    }

    // ----------------------------
    // Actual compile
    // ----------------------------

    CompiledKernel compile_internal(
        const std::string& source,
        const std::string& source_name,
        const std::string& kernel_name,
        const std::vector<std::string>& extra_options,
        const std::string& module_key,
        const std::string& code_key,
        int device_id = 0
    ) {
        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_internal: Entry\n");
        TNN_LOG_TRACE("[TRACE]   source_name: %s\n", source_name.c_str());
        TNN_LOG_TRACE("[TRACE]   kernel_name: %s\n", kernel_name.c_str());
        TNN_LOG_TRACE("[TRACE]   source size: %zu\n", source.size());
        TNN_LOG_TRACE("[TRACE]   headers: %zu\n", headers_.size());
        TNN_LOG_TRACE("[TRACE]   code_key: %s\n", code_key.c_str());
        TNN_LOG_TRACE("[TRACE]   module_key: %s\n", module_key.c_str());
        fflush(stdout);

        // Build header arrays for orortcCreateProgram
        std::vector<const char*> header_sources;
        std::vector<const char*> header_names;
        header_sources.reserve(headers_.size());
        header_names.reserve(headers_.size());

        for (const auto& h : headers_) {
            if (h.name == source_name) {
                printf("[WARNING] Skipping header '%s' - same as main source\n", h.name.c_str());
                continue;
            }
            TNN_LOG_TRACE("[TRACE]   Header: %s (size=%zu)\n", h.name.c_str(), h.source.size());
            header_sources.push_back(h.source.c_str());
            header_names.push_back(h.name.c_str());
        }

        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_internal: Calling orortcCreateProgram\n");
        TNN_LOG_TRACE("[TRACE]   Passing %d headers to HIPRTC\n", (int)header_sources.size());
        fflush(stdout);

        orortcProgram prog{};
        orortcResult rc{};

#if defined(_MSC_VER)
        __try {
            TNN_LOG_TRACE("[TRACE] About to call orortcCreateProgram...\n");
            fflush(stdout);

            rc = orortcCreateProgram(
                &prog,
                source.c_str(),
                source_name.c_str(),
                static_cast<int>(header_sources.size()),
                header_sources.empty() ? nullptr : header_sources.data(),
                header_names.empty() ? nullptr : header_names.data()
            );

            TNN_LOG_TRACE("[TRACE] orortcCreateProgram call returned successfully\n");
            fflush(stdout);
        }
        __except(EXCEPTION_EXECUTE_HANDLER) {
            printf("[ERROR] RTCCompiler: orortcCreateProgram crashed! Exception code: 0x%08X\n", (unsigned int)GetExceptionCode());
            fflush(stdout);
            throw std::runtime_error("orortcCreateProgram crashed with SEH exception");
        }
#else
        rc = orortcCreateProgram(
            &prog,
            source.c_str(),
            source_name.c_str(),
            static_cast<int>(header_sources.size()),
            header_sources.empty() ? nullptr : header_sources.data(),
            header_names.empty() ? nullptr : header_names.data()
        );
#endif

        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_internal: orortcCreateProgram returned %d\n", rc);
        fflush(stdout);

        if (rc != ORORTC_SUCCESS) {
            throw std::runtime_error("Failed to create HIPRTC program");
        }

        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_internal: HIPRTC program created, preparing compilation\n");
        fflush(stdout);

        // Normalize options (dedupe + deterministic ordering)
        const auto defaults = build_default_options();
        const auto norm = normalize_options(defaults, extra_options);

        // Convert to const char* (lifetime from std::string storage in norm.ordered)
        std::vector<const char*> options;
        options.reserve(norm.ordered.size());
        for (auto& s : norm.ordered) {
            options.push_back(s.c_str());
        }

        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_internal: Calling orortcCompileProgram with %d options\n",
               (int)options.size());
        for (size_t i = 0; i < options.size(); i++) {
            TNN_LOG_TRACE("[TRACE]   Option %zu: %s\n", i, options[i]);
        }
        fflush(stdout);

#if defined(_MSC_VER)
        __try {
            rc = orortcCompileProgram(
                prog,
                static_cast<int>(options.size()),
                options.empty() ? nullptr : options.data()
            );
        }
        __except(EXCEPTION_EXECUTE_HANDLER) {
            printf("[ERROR] RTCCompiler: orortcCompileProgram crashed! Exception code: 0x%08X\n", (unsigned int)GetExceptionCode());
            fflush(stdout);
            orortcDestroyProgram(&prog);
            throw std::runtime_error("orortcCompileProgram crashed with SEH exception");
        }
#else
        rc = orortcCompileProgram(
            prog,
            static_cast<int>(options.size()),
            options.empty() ? nullptr : options.data()
        );
#endif

        TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_internal: orortcCompileProgram returned %d\n", rc);
        fflush(stdout);

        if (rc != ORORTC_SUCCESS) {
            size_t log_size = 0;
            orortcGetProgramLogSize(prog, &log_size);
            std::vector<char> log(log_size ? log_size : 1, '\0');
            if (log_size > 0) {
                orortcGetProgramLog(prog, log.data());
            }
            orortcDestroyProgram(&prog);
            throw std::runtime_error(
                "HIPRTC compilation failed for " + source_name + ":\n" +
                std::string(log.data())
            );
        }

        size_t code_size = 0;
        orortcGetCodeSize(prog, &code_size);
        std::vector<char> code(code_size);
        orortcGetCode(prog, code.data());

        orortcDestroyProgram(&prog);

        // Save to AMD disk cache
        if (tnn_is_amd_device(device_id)) {
            auto code_options = filter_device_specific_options(norm.sorted);
            std::string disk_hash = make_disk_cache_hash(source, kernel_name, code_options);
            save_to_disk_cache(disk_hash, code);
        }

        CompiledKernel kernel;
        kernel.kernel_name = kernel_name;

        TNN_LOG_TRACE("[TRACE] About to load module (%zu bytes code)\n", code_size);

        size_t free_mem = 0, total_mem = 0;
        (void)oroMemGetInfo(&free_mem, &total_mem);
        TNN_LOG_TRACE("[TRACE] GPU memory: %zu MB free / %zu MB total\n",
               free_mem / (1024 * 1024), total_mem / (1024 * 1024));
        fflush(stdout);

        oroError_t err = oroModuleLoadData(&kernel.module, code.data());
        if (err != oroSuccess) {
            printf("[ERROR] oroModuleLoadData failed: %d (%s)\n", (int)err, tnn_error_string(err));
            (void)oroMemGetInfo(&free_mem, &total_mem);
            printf("[ERROR] GPU memory after failure: %zu MB free / %zu MB total\n",
                   free_mem / (1024 * 1024), total_mem / (1024 * 1024));
            fflush(stdout);
            throw std::runtime_error("oroModuleLoadData failed");
        }

        err = oroModuleGetFunction(&kernel.function, kernel.module, kernel.kernel_name.c_str());
        if (err != oroSuccess) {
            (void)oroModuleUnload(kernel.module);
            throw std::runtime_error("Failed to get kernel function: " + kernel_name);
        }

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // Store compiled code in code cache (shared across identical GPUs)
            CompiledCode cached_code;
            cached_code.code = std::move(code);
            cached_code.kernel_name = kernel_name;
            code_cache_[code_key] = std::move(cached_code);

            // Store loaded module in module cache (per-device)
            module_cache_[module_key] = kernel;

            TNN_LOG_TRACE("[TRACE] RTCCompiler::compile_internal: Cached code and module\n");
            TNN_LOG_TRACE("[TRACE]   code_key='%s'\n", code_key.c_str());
            TNN_LOG_TRACE("[TRACE]   module_key='%s'\n", module_key.c_str());
            fflush(stdout);
        }

        return kernel;
    }

private:
    std::string disk_cache_dir_;

    mutable std::mutex cache_mutex_;
    // Two-tier cache system:
    // 1. code_cache_: Stores compiled PTX/binary (slow compilation, shared across identical GPUs)
    // 2. module_cache_: Stores loaded oroModule_t (fast load, per-device context)
    std::unordered_map<std::string, CompiledCode> code_cache_;
    std::unordered_map<std::string, CompiledKernel> module_cache_;

    std::vector<Header> headers_;
};

extern "C" bool precompile_all_kernels();
