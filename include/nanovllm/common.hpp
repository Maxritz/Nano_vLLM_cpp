#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define NANOVLLM_USE_HIP 1
#define HIP_CHECK(expr)                                                \
  do {                                                                 \
    hipError_t hip_check_err_ = (expr);                                \
    if (hip_check_err_ != hipSuccess) {                                \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d\n",             \
                   (int)hip_check_err_, hipGetErrorString(hip_check_err_), \
                   __FILE__, __LINE__);                                \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)
#else
#define NANOVLLM_USE_HIP 0
#endif

inline float fp16_to_float(uint16_t h) {
  uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03ffu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    exp = exp - 15 + 127;
    bits = sign | (exp << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline float bf16_to_float(uint16_t h) {
  uint32_t bits = uint32_t(h) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// Round-to-nearest-even float -> half, matching numpy astype(np.float16)
// (overflow saturates to Inf). Used only by the GGUF host-upcast path.
inline uint16_t float_to_fp16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = int32_t((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;
  if (exp >= 31) return uint16_t(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return uint16_t(sign);
    mant |= 0x800000u;
    uint32_t t = uint32_t(14 - exp);
    uint32_t m = mant >> t;
    uint32_t dropped = mant & ((t >= 32 ? 0xffffffffu : ((1u << t) - 1u)));
    uint32_t half = 1u << (t - 1);
    if (dropped > half || (dropped == half && (m & 1u))) ++m;
    return uint16_t(sign | (m & 0x3ffu));
  }
  uint32_t m = mant >> 13;
  uint32_t dropped = mant & 0x1fffu;
  if (dropped > 0x1000u || (dropped == 0x1000u && (m & 1u))) {
    if (++m == 0x400u) {
      m = 0;
      if (++exp >= 31) return uint16_t(sign | 0x7c00u);
    }
  }
  return uint16_t(sign | (uint32_t(exp) << 10) | m);
}

#if NANOVLLM_USE_HIP

template <class T>
struct DevVec {
  T* d = nullptr;
  size_t n = 0;
  DevVec() = default;
  DevVec(const DevVec&) = delete;
  DevVec& operator=(const DevVec&) = delete;
  DevVec(DevVec&& o) noexcept : d(o.d), n(o.n) { o.d = nullptr; o.n = 0; }
  DevVec& operator=(DevVec&& o) noexcept {
    if (this != &o) { free(); d = o.d; n = o.n; o.d = nullptr; o.n = 0; }
    return *this;
  }
  explicit DevVec(const std::vector<T>& host) { assign(host); }
  ~DevVec() { free(); }
  void assign(const std::vector<T>& host) {
    free();
    n = host.size();
    if (n == 0) return;
    HIP_CHECK(hipMalloc(&d, n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d, host.data(), n * sizeof(T), hipMemcpyHostToDevice));
  }
  void free() {
    if (d) [[maybe_unused]] hipError_t ignored = hipFree(d);
    d = nullptr;
    n = 0;
  }
  T* operator*() const { return d; }
  T* operator->() const { return d; }
};

template <class T>
struct DeviceBuf {
  T* d = nullptr;
  size_t n = 0;
  DeviceBuf() = default;
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  DeviceBuf(DeviceBuf&& o) noexcept : d(o.d), n(o.n) { o.d = nullptr; o.n = 0; }
  DeviceBuf& operator=(DeviceBuf&& o) noexcept {
    if (this != &o) { reset(); d = o.d; n = o.n; o.d = nullptr; o.n = 0; }
    return *this;
  }
  ~DeviceBuf() { reset(); }
  void alloc(size_t count) {
    reset();
    n = count;
    if (n == 0) return;
    HIP_CHECK(hipMalloc(&d, n * sizeof(T)));
    HIP_CHECK(hipMemset(d, 0, n * sizeof(T)));
  }
  void reset() {
    if (d) [[maybe_unused]] hipError_t ignored = hipFree(d);
    d = nullptr;
    n = 0;
  }
  T* ptr() const { return d; }
  size_t size() const { return n; }
  operator T*() const { return d; }
};

inline float* malloc_device_float(size_t n) {
  if (n == 0) n = 1;
  float* p = nullptr;
  HIP_CHECK(hipMalloc(&p, n * sizeof(float)));
  return p;
}

inline int* malloc_device_int(size_t n) {
  if (n == 0) n = 1;
  int* p = nullptr;
  HIP_CHECK(hipMalloc(&p, n * sizeof(int)));
  return p;
}

inline int64_t* malloc_device_i64(size_t n) {
  if (n == 0) n = 1;
  int64_t* p = nullptr;
  HIP_CHECK(hipMalloc(&p, n * sizeof(int64_t)));
  return p;
}

#endif  // NANOVLLM_USE_HIP
