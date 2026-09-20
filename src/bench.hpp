#ifndef VIBEVOICE_BENCH_HPP
#define VIBEVOICE_BENCH_HPP

// Per-phase wall-clock accounting for the TTS loop, printed when the
// VIBEVOICE_BENCH environment variable is set. Zero cost otherwise beyond
// one env lookup per process.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace vv {

inline bool bench_enabled() {
    static const bool on = std::getenv("VIBEVOICE_BENCH") != nullptr;
    return on;
}

struct BenchClock {
    using clock = std::chrono::steady_clock;
    clock::time_point t0 = clock::now();
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// Accumulates (name -> total ms, count). Print with report().
struct BenchTotals {
    struct Row { std::string name; double ms = 0; int n = 0; };
    std::vector<Row> rows;

    void add(const char* name, double ms) {
        for (auto& r : rows) if (r.name == name) { r.ms += ms; r.n++; return; }
        rows.push_back({name, ms, 1});
    }
    void report(const char* title) const {
        if (!bench_enabled()) return;
        std::fprintf(stderr, "[bench] %s\n", title);
        for (const auto& r : rows) {
            std::fprintf(stderr, "[bench]   %-22s %9.1f ms  (%d calls, %.2f ms/call)\n",
                         r.name.c_str(), r.ms, r.n, r.n ? r.ms / r.n : 0.0);
        }
    }
};

// Scoped timer: adds elapsed ms to `totals[name]` on destruction.
struct BenchScope {
    BenchTotals* totals;
    const char*  name;
    BenchClock   clk;
    BenchScope(BenchTotals* t, const char* n) : totals(t), name(n) {}
    ~BenchScope() { if (totals && bench_enabled()) totals->add(name, clk.ms()); }
};

}  // namespace vv

#endif  // VIBEVOICE_BENCH_HPP
