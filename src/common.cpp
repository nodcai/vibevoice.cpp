#include "common.hpp"

#include "ggml.h"

#include <atomic>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <mutex>
#include <sys/stat.h>

namespace vv {

namespace {
struct LogState {
    std::mutex   mu;
    vv_log_cb    cb       = nullptr;
    void*        user     = nullptr;
    vv_log_level min_lvl  = VV_LOG_INFO;
};
LogState& state() {
    static LogState s;
    return s;
}

// ggml logs to stderr on its own. That is wrong for a library: a host that
// installed a callback still gets "compiling pipeline: kernel_..." the first
// time Metal builds a kernel, and with the streaming API that lands in the
// middle of a reply. Route ggml through the same callback. It has to be this
// library's ggml_log_set: a host that links its own ggml, or several, cannot
// reach a statically linked copy with hidden symbols from outside.
vv_log_level ggml_to_vv(ggml_log_level l) {
    switch (l) {
        case GGML_LOG_LEVEL_ERROR: return VV_LOG_ERROR;
        case GGML_LOG_LEVEL_WARN:  return VV_LOG_WARN;
        case GGML_LOG_LEVEL_DEBUG: return VV_LOG_DEBUG;
        default:                   return VV_LOG_INFO;   // INFO and CONT
    }
}

void ggml_log_forward(ggml_log_level glvl, const char* text, void*) {
    auto& s = state();
    const vv_log_level lvl = ggml_to_vv(glvl);
    if (lvl > s.min_lvl) return;
    vv_log_cb cb;
    void*     user;
    {
        // Copy under the lock, call outside it: the callback may log again.
        std::lock_guard<std::mutex> lk(s.mu);
        cb   = s.cb;
        user = s.user;
    }
    if (cb) cb(lvl, text, user);      // ggml's text already ends in "\n"
    else    std::fputs(text, stderr);
}

const char* lvl_str(vv_log_level l) {
    switch (l) {
        case VV_LOG_ERROR: return "E";
        case VV_LOG_WARN:  return "W";
        case VV_LOG_INFO:  return "I";
        case VV_LOG_DEBUG: return "D";
    }
    return "?";
}
}  // namespace

void log(vv_log_level lvl, const char* fmt, ...) {
    auto& s = state();
    if (lvl > s.min_lvl) return;

    // One complete line per call, newline included, in both paths: a host
    // callback can write the text straight out, and ggml's messages (which
    // arrive newline-terminated already) look the same to it.
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buf, sizeof(buf) - 2, fmt, ap);
    va_end(ap);
    size_t len = n < 0 ? 0 : std::min(static_cast<size_t>(n), sizeof(buf) - 2);
    buf[len++] = '\n';
    buf[len]   = '\0';

    vv_log_cb cb;
    void*     user;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        cb   = s.cb;
        user = s.user;
    }
    if (cb) {
        cb(lvl, buf, user);
    } else {
        std::fprintf(lvl <= VV_LOG_WARN ? stderr : stdout, "[vv %s] %s", lvl_str(lvl), buf);
    }
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

size_t file_size(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return 0;
    return static_cast<size_t>(st.st_size);
}

bool read_file(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    auto n = f.tellg();
    if (n < 0) return false;
    f.seekg(0, std::ios::beg);
    out->resize(static_cast<size_t>(n));
    if (n > 0) f.read(out->data(), n);
    return f.good() || f.eof();
}

}  // namespace vv

extern "C" {

void vv_set_log_callback(vv_log_cb cb, void* user_data) {
    {
        auto& s = vv::state();
        std::lock_guard<std::mutex> lk(s.mu);
        s.cb   = cb;
        s.user = user_data;
    }
    // Outside the lock: ggml may log from inside ggml_log_set, and the
    // forwarder takes the same mutex.
    ggml_log_set(vv::ggml_log_forward, nullptr);
}

const char* vv_version(void) {
    return "vibevoice.cpp 0.0.1 (M1-foundations)";
}

}  // extern "C"
