#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <thread>
#include <atomic>
#include <map>
#include <functional>
#include <memory>
#include <signal.h>
#include <mutex>

#include "terminal.h"

#ifdef __x86_64__
#include <cpuid.h>

#ifdef _WIN32
#include <Windows.h>

class WinRing0Dynamic {
private:
    HMODULE hDll = nullptr;
    
public:
    // Function pointers
    BOOL (WINAPI *InitializeOls)() = nullptr;
    VOID (WINAPI *DeinitializeOls)() = nullptr;
    BOOL (WINAPI *RdmsrTx)(DWORD, PDWORD, PDWORD, DWORD_PTR) = nullptr;
    BOOL (WINAPI *WrmsrTx)(DWORD, DWORD, DWORD, DWORD_PTR) = nullptr;
    
    WinRing0Dynamic() {
        hDll = LoadLibraryA("WinRing0x64.dll");
        if (hDll) {
            *(FARPROC*)&InitializeOls = GetProcAddress(hDll, "InitializeOls");
            *(FARPROC*)&DeinitializeOls = GetProcAddress(hDll, "DeinitializeOls");
            *(FARPROC*)&RdmsrTx = GetProcAddress(hDll, "RdmsrTx");  
            *(FARPROC*)&WrmsrTx = GetProcAddress(hDll, "WrmsrTx");
        }
    }
    
    ~WinRing0Dynamic() {
        if (hDll) FreeLibrary(hDll);
    }
    
    bool isLoaded() const {
        return hDll && InitializeOls && RdmsrTx && WrmsrTx;
    }
};

static WinRing0Dynamic g_winRing0;

inline bool initMSRAccess() {
    if (!g_winRing0.isLoaded()) {
        return false;
    }
    return g_winRing0.InitializeOls() != 0; // TRUE means success
}

inline bool readMSR(uint32_t reg, uint64_t& value, int core = 0) {
    if (!g_winRing0.isLoaded()) {
        return false;
    }
    
    DWORD eax = 0, edx = 0;
    if (g_winRing0.RdmsrTx(reg, &eax, &edx, (1ULL << core))) {
        value = ((uint64_t)edx << 32) | eax;
        return true;
    }
    return false;
}

inline bool writeMSR(uint32_t reg, uint64_t value, int core = 0) {
    if (!g_winRing0.isLoaded()) {
        return false;
    }
    
    DWORD eax = static_cast<DWORD>(value & 0xFFFFFFFF);
    DWORD edx = static_cast<DWORD>(value >> 32);
    return g_winRing0.WrmsrTx(reg, eax, edx, (1ULL << core)) != 0;
}

#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#endif
#endif

// CPU detection (architecture-protected)
enum class CPUType {
    UNSUPPORTED,
    INTEL,
    AMD_ZEN1_ZEN2,
    AMD_ZEN3,
    AMD_ZEN4_ZEN5
};

// Base MSR optimization profile class
class MSROptimizationProfile {
public:
    virtual ~MSROptimizationProfile() = default;
    
    virtual std::string getName() const = 0;
    virtual std::vector<std::pair<uint32_t, uint64_t>> getMSRValues(CPUType cpuType) const = 0;
};

// RandomX-specific profile implementation
class RandomXOptimizationProfile : public MSROptimizationProfile {
public:
    std::string getName() const override {
        return "RandomX";
    }
    
    std::vector<std::pair<uint32_t, uint64_t>> getMSRValues(CPUType cpuType) const override {
        std::vector<std::pair<uint32_t, uint64_t>> msrValues;
        
        switch (cpuType) {
            case CPUType::INTEL:
                msrValues.emplace_back(0x1A4, 0xF);
                break;
                
            case CPUType::AMD_ZEN1_ZEN2:
                msrValues.emplace_back(0xC0011020, 0x0);
                msrValues.emplace_back(0xC0011021, 0x40);
                msrValues.emplace_back(0xC0011022, 0x1510000);
                msrValues.emplace_back(0xC001102B, 0x2000CC16);
                break;
                
            case CPUType::AMD_ZEN3:
                msrValues.emplace_back(0xC0011020, 0x4480000000000);
                msrValues.emplace_back(0xC0011021, 0x1C000200000040);
                msrValues.emplace_back(0xC0011022, 0xC000000401570000);
                msrValues.emplace_back(0xC001102B, 0x2000CC10);
                break;
                
            case CPUType::AMD_ZEN4_ZEN5:
                msrValues.emplace_back(0xC0011020, 0x4400000000000);
                msrValues.emplace_back(0xC0011021, 0x4000000000040);
                msrValues.emplace_back(0xC0011022, 0x8680000401570000);
                msrValues.emplace_back(0xC001102B, 0x2040CC10);
                break;
                
            default:
                break;
        }
        
        return msrValues;
    }
};

// Example: Ethash optimization profile (placeholder values - replace with actual research)
class EthashOptimizationProfile : public MSROptimizationProfile {
public:
    std::string getName() const override {
        return "Ethash";
    }
    
    std::vector<std::pair<uint32_t, uint64_t>> getMSRValues(CPUType cpuType) const override {
        std::vector<std::pair<uint32_t, uint64_t>> msrValues;
        
        // Example values - would need real research for actual Ethash optimizations
        if (cpuType == CPUType::INTEL) {
            msrValues.emplace_back(0x1A4, 0x3); // Different from RandomX
        }
        
        return msrValues;
    }
};

// Main MSR Manager class
class MSRManager {
private:
    struct MSRBackup {
        uint32_t reg;
        std::vector<uint64_t> originalValues;
    };
    
    CPUType cpuType = CPUType::UNSUPPORTED;
    int numCores = 0;
    bool msrAvailable = false;
    std::atomic<bool> optimizationsApplied{false};
    std::string activeProfileName;
    std::vector<MSRBackup> backups;
    std::map<std::string, std::shared_ptr<MSROptimizationProfile>> profiles;
    std::mutex mutex;

public:
    MSRManager() {
        #ifdef __x86_64__
        detectCPU();
        numCores = std::thread::hardware_concurrency();
        msrAvailable = initMSRAccess();
        
        if (msrAvailable) {
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nMSR access available. Detected: ";
            switch (cpuType) {
                case CPUType::INTEL:
                    std::cout << "Intel CPU";
                    break;
                case CPUType::AMD_ZEN1_ZEN2:
                    std::cout << "AMD Zen1/Zen2 CPU";
                    break;
                case CPUType::AMD_ZEN3:
                    std::cout << "AMD Zen3 CPU";
                    break;
                case CPUType::AMD_ZEN4_ZEN5:
                    std::cout << "AMD Zen4/Zen5 CPU";
                    break;
                default:
                    std::cout << "Unsupported CPU";
            }
            std::cout << " with " << numCores << " logical cores" << std::endl;
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        } else {
            setcolor(RED);
            std::cerr << "MSR access not available. Run as administrator/root." << std::endl;
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        }
        
        // Register default profiles
        registerProfile(std::make_shared<RandomXOptimizationProfile>());
        registerProfile(std::make_shared<EthashOptimizationProfile>());
        #else
        setcolor(RED);
        std::cout << "MSR optimization not available on non-x86_64 architecture" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        #endif
    }
    
    ~MSRManager() {
        restoreOriginalValues();
    }
    
    void registerProfile(std::shared_ptr<MSROptimizationProfile> profile) {
        std::lock_guard<std::mutex> lock(mutex);
        profiles[profile->getName()] = profile;
    }
    
    bool applyOptimizationProfile(const std::string& profileName) {
        #ifdef __x86_64__
        if (!msrAvailable) {
            std::cerr << "MSR access not available" << std::endl;
            return false;
        }
        
        if (optimizationsApplied.load()) {
            std::cout << "Another optimization profile is already active. Restoring first..." << std::endl;
            restoreOriginalValues();
        }
        
        std::shared_ptr<MSROptimizationProfile> profile;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = profiles.find(profileName);
            if (it == profiles.end()) {
                std::cerr << "Optimization profile '" << profileName << "' not found" << std::endl;
                return false;
            }
            profile = it->second;
        }
        
        std::vector<std::pair<uint32_t, uint64_t>> msrValues = profile->getMSRValues(cpuType);
        
        if (msrValues.empty()) {
            std::cerr << "No MSR values defined for this CPU in profile: " << profileName << std::endl;
            return false;
        }
        
        // Backup current MSR values before modifying
        backupMSRValues(msrValues);
        
        // Apply new MSR values to all cores
        bool success = true;
        for (const auto& [reg, value] : msrValues) {
            for (int core = 0; core < numCores; core++) {
                if (!writeMSR(reg, value, core)) {
                    std::cerr << "Failed to write MSR 0x" << std::hex << reg 
                              << " on core " << std::dec << core << std::endl;
                    success = false;
                }
            }
        }
        
        if (success) {
            setcolor(BRIGHT_YELLOW);
            std::cout << profileName << " MSR optimizations applied successfully" << std::endl;
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
            activeProfileName = profileName;
            optimizationsApplied.store(true);
        } else {
            std::cerr << "MSR optimization partially failed - system may be in inconsistent state" << std::endl;
            restoreOriginalValues();
        }
        
        return success;
        #else
        return false;
        #endif
    }
    
    bool restoreOriginalValues() {
        #ifdef __x86_64__
        std::lock_guard<std::mutex> lock(mutex);
        
        if (!msrAvailable || !optimizationsApplied.load() || backups.empty()) 
            return false;
        
        bool success = true;
        for (const auto& backup : backups) {
            for (int core = 0; core < backup.originalValues.size(); core++) {
                if (!writeMSR(backup.reg, backup.originalValues[core], core)) {
                    std::cerr << "Failed to restore MSR 0x" << std::hex << backup.reg 
                              << " on core " << std::dec << core << std::endl;
                    success = false;
                }
            }
        }
        
        if (success) {
            std::cout << "Original MSR values restored successfully" << std::endl;
            optimizationsApplied.store(false);
            activeProfileName = "";
            backups.clear();
        }
        
        return success;
        #else
        return false;
        #endif
    }
    
    std::string getActiveProfileName() const {
        return activeProfileName;
    }
    
    bool isOptimizationActive() const {
        return optimizationsApplied.load();
    }
    
    CPUType getCPUType() const {
        return cpuType;
    }

private:
    #ifdef __x86_64__
    void detectCPU() {
        uint32_t eax, ebx, ecx, edx;
        
        // Get vendor
        if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
            char vendor[13] = {0};
            memcpy(vendor, &ebx, 4);
            memcpy(vendor + 4, &edx, 4);
            memcpy(vendor + 8, &ecx, 4);
            vendor[12] = '\0';
            
            std::string vendorStr(vendor);
            
            if (vendorStr.find("Intel") != std::string::npos) {
                cpuType = CPUType::INTEL;
            } else if (vendorStr.find("AMD") != std::string::npos) {
                // Get detailed AMD info
                if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                    int family = ((eax >> 8) & 0xF) + ((eax >> 20) & 0xFF);
                    int model = ((eax >> 4) & 0xF) | ((eax >> 12) & 0xF0);
                    
                    if (family == 25) {
                        if (model == 97) {
                            cpuType = CPUType::AMD_ZEN4_ZEN5;
                        } else {
                            cpuType = CPUType::AMD_ZEN3;
                        }
                    } else if (family == 26) {
                        cpuType = CPUType::AMD_ZEN4_ZEN5;
                    } else if (family == 23) {
                        cpuType = CPUType::AMD_ZEN1_ZEN2;
                    }
                }
            }
        }
    }
    
    bool initMSRAccess() {
        #ifdef _WIN32
        return ::initMSRAccess();
        #else
        std::ifstream msrCheck("/dev/cpu/0/msr");
        if (!msrCheck.good()) {
            // Try loading msr module if not already loaded
            system("modprobe msr allow_writes=on 2>/dev/null || true");
            system("echo on > /sys/module/msr/parameters/allow_writes 2>/dev/null || true");
            msrCheck.close();
            msrCheck.open("/dev/cpu/0/msr");
        }
        return msrCheck.good();
        #endif
    }
    
    bool readMSR(uint32_t reg, uint64_t& value, int core = 0) {
        #ifdef _WIN32
        return ::readMSR(reg, value, core);
        return false;
        #else
        char path[64];
        snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
        int fd = open(path, O_RDONLY);
        if (fd < 0) return false;
        
        bool result = pread(fd, &value, sizeof(value), reg) == sizeof(value);
        close(fd);
        return result;
        #endif
    }
    
    bool writeMSR(uint32_t reg, uint64_t value, int core = 0) {
        #ifdef _WIN32
        return ::writeMSR(reg, value, core);
        #else
        char path[64];
        snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
        int fd = open(path, O_WRONLY);
        if (fd < 0) return false;
        
        bool result = pwrite(fd, &value, sizeof(value), reg) == sizeof(value);
        close(fd);
        return result;
        #endif
    }
    
    void backupMSRValues(const std::vector<std::pair<uint32_t, uint64_t>>& msrList) {
        backups.clear();
        
        for (const auto& [reg, _] : msrList) {
            MSRBackup backup;
            backup.reg = reg;
            backup.originalValues.resize(numCores);
            
            for (int core = 0; core < numCores; core++) {
                uint64_t value = 0;
                if (readMSR(reg, value, core)) {
                    backup.originalValues[core] = value;
                } else {
                    std::cerr << "Failed to read MSR 0x" << std::hex << reg 
                              << " on core " << std::dec << core << std::endl;
                    // Use default value of 0 on read failure
                }
            }
            
            backups.push_back(backup);
        }
    }
    #endif
};

// Global singleton
class MSRManagerGlobal {
private:
    static std::unique_ptr<MSRManager> instance;
    static std::mutex instanceMutex;
    static std::atomic<bool> signalHandlersRegistered;
    
    static void signalHandler(int sig) {
        std::cout << "Caught signal " << sig << ", restoring MSR values and exiting..." << std::endl;
        if (instance) {
            if (instance->isOptimizationActive()) {
                std::string profile = instance->getActiveProfileName();
                instance->restoreOriginalValues();
                std::cout << "MSR values for profile '" << profile << "' restored" << std::endl;
            }
        }
        exit(sig);
    }
    
    static void registerSignalHandlers() {
        if (!signalHandlersRegistered.load()) {
            signal(SIGINT, signalHandler);
            signal(SIGTERM, signalHandler);
            signalHandlersRegistered.store(true);
        }
    }

public:
    static MSRManager* getInstance() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (!instance) {
            instance = std::make_unique<MSRManager>();
            registerSignalHandlers();
        }
        return instance.get();
    }
    
    static void cleanup() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (instance && instance->isOptimizationActive()) {
            instance->restoreOriginalValues();
        }
        instance.reset();
    }
};

std::unique_ptr<MSRManager> MSRManagerGlobal::instance = nullptr;
std::mutex MSRManagerGlobal::instanceMutex;
std::atomic<bool> MSRManagerGlobal::signalHandlersRegistered{false};

// Simple interface for algorithm-specific optimizations
inline bool applyMSROptimization(const std::string& algorithm) {
    MSRManager* manager = MSRManagerGlobal::getInstance();
    return manager->applyOptimizationProfile(algorithm);
}

inline bool resetMSROptimizations() {
    MSRManager* manager = MSRManagerGlobal::getInstance();
    return manager->restoreOriginalValues();
}

inline void cleanupMSROnExit() {
    MSRManagerGlobal::cleanup();
}