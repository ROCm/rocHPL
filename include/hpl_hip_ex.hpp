
#ifndef HPL_HIP_EX
#define HPL_HIP_EX

#include "hip/hip_runtime.h"

#if defined(__GFX8__) || defined(__GFX9__)
__device__ static constexpr int WarpSize = 64;
#else
__device__ static constexpr int WarpSize = 32;
#endif

namespace hip_ex {

enum thread_scope {
  thread_scope_system = __HIP_MEMORY_SCOPE_SYSTEM,
  thread_scope_device = __HIP_MEMORY_SCOPE_AGENT,
  thread_scope_block  = __HIP_MEMORY_SCOPE_WORKGROUP,
  thread_scope_wave   = __HIP_MEMORY_SCOPE_WAVEFRONT,
  thread_scope_thread = __HIP_MEMORY_SCOPE_SINGLETHREAD
};

enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_consume = __ATOMIC_CONSUME,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
};

template <typename T, thread_scope scope>
struct __atomic_ref_common {
  T* _ptr;

  inline constexpr __atomic_ref_common(T& ref) : _ptr(&ref) {}

  __host__ __device__ inline void store(
      T            val,
      memory_order order = memory_order_seq_cst) noexcept {
    __hip_atomic_store(_ptr, val, order, scope);
  }
  __host__ __device__ inline T load(
      memory_order order = memory_order_seq_cst) const noexcept {
    return __hip_atomic_load(_ptr, order, scope);
  }
  __host__ __device__ inline   operator T() const noexcept { return load(); }
  __host__ __device__ inline T exchange(
      T            val,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_exchange(&_ptr, val, order, scope);
  }
  __host__ __device__ inline bool compare_exchange_weak(
      T&           expected,
      T            desired,
      memory_order success_order,
      memory_order failure_order) noexcept {
    return __hip_atomic_compare_exchange_weak(
        _ptr, &expected, desired, success_order, failure_order, scope);
  }
  __host__ __device__ inline bool compare_exchange_strong(
      T&           expected,
      T            desired,
      memory_order success_order,
      memory_order failure_order) noexcept {
    return __hip_atomic_compare_exchange_strong(
        _ptr, &expected, desired, success_order, failure_order, scope);
  }
  __host__ __device__ inline bool compare_exchange_weak(
      T&           expected,
      T            desired,
      memory_order order = memory_order_seq_cst) noexcept {
    if(memory_order_acq_rel == order)
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, memory_order_acquire, scope);
    else if(memory_order_release == order)
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, memory_order_relaxed, scope);
    else
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, order, scope);
  }
  __host__ __device__ inline bool compare_exchange_strong(
      T&           expected,
      T            desired,
      memory_order order = memory_order_seq_cst) noexcept {
    if(memory_order_acq_rel == order)
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, memory_order_acquire, scope);
    else if(memory_order_release == order)
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, memory_order_relaxed, scope);
    else
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, order, scope);
  }
  __host__ __device__ inline void wait(
      T            val,
      memory_order order = memory_order_seq_cst) const noexcept {
    while(load(order) == val) {};
  }
};

template <typename T, thread_scope scope>
struct __atomic_ref_arithmetic {
  T* _ptr;

  inline constexpr __atomic_ref_arithmetic(T& ref) : _ptr(&ref) {}

  __host__ __device__ inline void store(
      T            val,
      memory_order order = memory_order_seq_cst) noexcept {
    __hip_atomic_store(_ptr, val, order, scope);
  }
  __host__ __device__ inline T load(
      memory_order order = memory_order_seq_cst) const noexcept {
    return __hip_atomic_load(_ptr, order, scope);
  }
  __host__ __device__ inline   operator T() const noexcept { return load(); }
  __host__ __device__ inline T exchange(
      T            val,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_exchange(&_ptr, val, order, scope);
  }
  __host__ __device__ inline bool compare_exchange_weak(
      T&           expected,
      T            desired,
      memory_order success_order,
      memory_order failure_order) noexcept {
    return __hip_atomic_compare_exchange_weak(
        _ptr, &expected, desired, success_order, failure_order, scope);
  }
  __host__ __device__ inline bool compare_exchange_strong(
      T&           expected,
      T            desired,
      memory_order success_order,
      memory_order failure_order) noexcept {
    return __hip_atomic_compare_exchange_strong(
        _ptr, &expected, desired, success_order, failure_order, scope);
  }
  __host__ __device__ inline bool compare_exchange_weak(
      T&           expected,
      T            desired,
      memory_order order = memory_order_seq_cst) noexcept {
    if(memory_order_acq_rel == order)
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, memory_order_acquire, scope);
    else if(memory_order_release == order)
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, memory_order_relaxed, scope);
    else
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, order, scope);
  }
  __host__ __device__ inline bool compare_exchange_strong(
      T&           expected,
      T            desired,
      memory_order order = memory_order_seq_cst) noexcept {
    if(memory_order_acq_rel == order)
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, memory_order_acquire, scope);
    else if(memory_order_release == order)
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, memory_order_relaxed, scope);
    else
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, order, scope);
  }
  __host__ __device__ inline void wait(
      T            val,
      memory_order order = memory_order_seq_cst) const noexcept {
    while(load(order) == val) {};
  }

  __host__ __device__ inline T fetch_add(
      const T&     arg,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_add(_ptr, arg, order, scope);
  }

  __host__ __device__ inline T fetch_sub(
      const T&     arg,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_add(_ptr, -arg, order, scope);
  }

  __host__ __device__ inline T operator+=(T __op) noexcept {
    return fetch_add(__op) + __op;
  }
  __host__ __device__ inline T operator-=(T __op) noexcept {
    return fetch_sub(__op) - __op;
  }
};

template <typename T, thread_scope scope>
struct __atomic_ref_bitwise {
  T* _ptr;

  inline constexpr __atomic_ref_bitwise(T& ref) : _ptr(&ref) {}

  __host__ __device__ inline void store(
      T            val,
      memory_order order = memory_order_seq_cst) noexcept {
    __hip_atomic_store(_ptr, val, order, scope);
  }
  __host__ __device__ inline T load(
      memory_order order = memory_order_seq_cst) const noexcept {
    return __hip_atomic_load(_ptr, order, scope);
  }
  __host__ __device__ inline   operator T() const noexcept { return load(); }
  __host__ __device__ inline T exchange(
      T            val,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_exchange(&_ptr, val, order, scope);
  }
  __host__ __device__ inline bool compare_exchange_weak(
      T&           expected,
      T            desired,
      memory_order success_order,
      memory_order failure_order) noexcept {
    return __hip_atomic_compare_exchange_weak(
        _ptr, &expected, desired, success_order, failure_order, scope);
  }
  __host__ __device__ inline bool compare_exchange_strong(
      T&           expected,
      T            desired,
      memory_order success_order,
      memory_order failure_order) noexcept {
    return __hip_atomic_compare_exchange_strong(
        _ptr, &expected, desired, success_order, failure_order, scope);
  }
  __host__ __device__ inline bool compare_exchange_weak(
      T&           expected,
      T            desired,
      memory_order order = memory_order_seq_cst) noexcept {
    if(memory_order_acq_rel == order)
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, memory_order_acquire, scope);
    else if(memory_order_release == order)
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, memory_order_relaxed, scope);
    else
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, order, scope);
  }
  __host__ __device__ inline bool compare_exchange_strong(
      T&           expected,
      T            desired,
      memory_order order = memory_order_seq_cst) noexcept {
    if(memory_order_acq_rel == order)
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, memory_order_acquire, scope);
    else if(memory_order_release == order)
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, memory_order_relaxed, scope);
    else
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, order, scope);
  }
  __host__ __device__ inline void wait(
      T            val,
      memory_order order = memory_order_seq_cst) const noexcept {
    while(load(order) == val) {};
  }

  __host__ __device__ inline T fetch_add(
      const T&     arg,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_add(_ptr, arg, order, scope);
  }

  __host__ __device__ inline T fetch_sub(
      const T&     arg,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_add(_ptr, -arg, order, scope);
  }

  __host__ __device__ inline T operator+=(T __op) noexcept {
    return fetch_add(__op) + __op;
  }
  __host__ __device__ inline T operator-=(T __op) noexcept {
    return fetch_sub(__op) - __op;
  }

  __host__ __device__ inline T operator++(int) noexcept {
    return fetch_add(T(1));
  }
  __host__ __device__ inline T operator--(int) noexcept {
    return fetch_sub(T(1));
  }
  __host__ __device__ inline T operator++() noexcept {
    return fetch_add(T(1)) + T(1);
  }
  __host__ __device__ inline T operator--() noexcept {
    return fetch_sub(T(1)) - T(1);
  }

  __host__ __device__ inline T fetch_max(
      const T&     arg,
      memory_order order = memory_order_seq_cst) const noexcept {
    return __hip_atomic_fetch_max(_ptr, arg, order, scope);
  }

  __host__ __device__ inline T fetch_min(
      const T&     arg,
      memory_order order = memory_order_seq_cst) const noexcept {
    return __hip_atomic_fetch_min(_ptr, arg, order, scope);
  }

  __host__ __device__ inline T fetch_and(
      T            arg,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_and(&_ptr, arg, order, scope);
  }
  __host__ __device__ inline T fetch_or(
      T            arg,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_or(&_ptr, arg, order, scope);
  }
  __host__ __device__ inline T fetch_xor(
      T            arg,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_xor(&_ptr, arg, order, scope);
  }
  __host__ __device__ inline T operator&=(T arg) noexcept {
    return fetch_and(arg) & arg;
  }
  __host__ __device__ inline T operator|=(T arg) noexcept {
    return fetch_or(arg) | arg;
  }
  __host__ __device__ inline T operator^=(T arg) noexcept {
    return fetch_xor(arg) ^ arg;
  }
};

template <typename T, thread_scope scope>
struct __atomic_ref_pointer {
  T* _ptr;

  inline constexpr __atomic_ref_pointer(T& ref) : _ptr(&ref) {}

  __host__ __device__ inline void store(
      T            val,
      memory_order order = memory_order_seq_cst) noexcept {
    __hip_atomic_store(_ptr, val, order, scope);
  }
  __host__ __device__ inline T load(
      memory_order order = memory_order_seq_cst) const noexcept {
    return __hip_atomic_load(_ptr, order, scope);
  }
  __host__ __device__ inline   operator T() const noexcept { return load(); }
  __host__ __device__ inline T exchange(
      T            val,
      memory_order order = memory_order_seq_cst) noexcept {
    return __hip_atomic_exchange(&_ptr, val, order, scope);
  }
  __host__ __device__ inline bool compare_exchange_weak(
      T&           expected,
      T            desired,
      memory_order success_order,
      memory_order failure_order) noexcept {
    return __hip_atomic_compare_exchange_weak(
        _ptr, &expected, desired, success_order, failure_order, scope);
  }
  __host__ __device__ inline bool compare_exchange_strong(
      T&           expected,
      T            desired,
      memory_order success_order,
      memory_order failure_order) noexcept {
    return __hip_atomic_compare_exchange_strong(
        _ptr, &expected, desired, success_order, failure_order, scope);
  }
  __host__ __device__ inline bool compare_exchange_weak(
      T&           expected,
      T            desired,
      memory_order order = memory_order_seq_cst) noexcept {
    if(memory_order_acq_rel == order)
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, memory_order_acquire, scope);
    else if(memory_order_release == order)
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, memory_order_relaxed, scope);
    else
      return __hip_atomic_compare_exchange_weak(
          _ptr, &expected, desired, order, order, scope);
  }
  __host__ __device__ inline bool compare_exchange_strong(
      T&           expected,
      T            desired,
      memory_order order = memory_order_seq_cst) noexcept {
    if(memory_order_acq_rel == order)
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, memory_order_acquire, scope);
    else if(memory_order_release == order)
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, memory_order_relaxed, scope);
    else
      return __hip_atomic_compare_exchange_strong(
          _ptr, &expected, desired, order, order, scope);
  }
  __host__ __device__ inline void wait(
      T            val,
      memory_order order = memory_order_seq_cst) const noexcept {
    while(load(order) == val) {};
  }

  __host__ __device__ inline T fetch_add(
      const ptrdiff_t arg,
      memory_order    order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_add(_ptr, arg, order, scope);
  }

  __host__ __device__ inline T fetch_sub(
      const ptrdiff_t arg,
      memory_order    order = memory_order_seq_cst) noexcept {
    return __hip_atomic_fetch_add(_ptr, -arg, order, scope);
  }

  __host__ __device__ inline T operator+=(ptrdiff_t __op) noexcept {
    return fetch_add(__op) + __op;
  }
  __host__ __device__ inline T operator-=(ptrdiff_t __op) noexcept {
    return fetch_sub(__op) - __op;
  }

  __host__ __device__ inline T operator++(int) noexcept {
    return fetch_add(T(1));
  }
  __host__ __device__ inline T operator--(int) noexcept {
    return fetch_sub(T(1));
  }
  __host__ __device__ inline T operator++() noexcept {
    return fetch_add(T(1)) + T(1);
  }
  __host__ __device__ inline T operator--() noexcept {
    return fetch_sub(T(1)) - T(1);
  }

  __host__ __device__ inline T fetch_max(
      const T&     arg,
      memory_order order = memory_order_seq_cst) const noexcept {
    return __hip_atomic_fetch_max(_ptr, arg, order, scope);
  }

  __host__ __device__ inline T fetch_min(
      const T&     arg,
      memory_order order = memory_order_seq_cst) const noexcept {
    return __hip_atomic_fetch_min(_ptr, arg, order, scope);
  }
};

template <bool>
struct _IfImpl;

template <>
struct _IfImpl<true> {
  template <class _IfRes, class _ElseRes>
  using _Select = _IfRes;
};

template <>
struct _IfImpl<false> {
  template <class _IfRes, class _ElseRes>
  using _Select = _ElseRes;
};

template <bool _Cond, class _IfRes, class _ElseRes>
using _If = typename _IfImpl<_Cond>::template _Select<_IfRes, _ElseRes>;

template <typename T, thread_scope scope = thread_scope_system>
using __atomic_ref_impl = _If<std::is_pointer<T>::value,
                              __atomic_ref_pointer<T, scope>,
                              _If<std::is_floating_point<T>::value,
                                  __atomic_ref_arithmetic<T, scope>,
                                  _If<std::is_integral<T>::value,
                                      __atomic_ref_bitwise<T, scope>,
                                      __atomic_ref_common<T, scope>>>>;

template <class T, thread_scope scope>
class atomic_ref : public __atomic_ref_impl<T, scope> {
  public:
  using value_type                            = T;
  static constexpr size_t required_alignment  = sizeof(T);
  static constexpr bool   is_always_lock_free = sizeof(T) <= 8;

  explicit constexpr atomic_ref(T& ref) : __atomic_ref_impl<T, scope>(ref) {}

  T operator=(T v) const noexcept {
    this->store(v);
    return v;
  }

  atomic_ref(const atomic_ref&) noexcept         = default;
  atomic_ref& operator=(const atomic_ref&)       = delete;
  atomic_ref& operator=(const atomic_ref&) const = delete;

  __host__ __device__ inline bool is_lock_free() const noexcept {
    return (sizeof(T) <= 8);
  }
};

__device__ inline void atomic_thread_fence_system(memory_order order) {
  switch(order) {
    case memory_order_acquire:
    case memory_order_consume:
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      break;
    case memory_order_release:
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      break;
    case memory_order_acq_rel:
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
      break;
    case memory_order_seq_cst:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
      break;
    case memory_order_relaxed: break;
  };
}

__device__ inline void atomic_thread_fence_device(memory_order order) {
  switch(order) {
    case memory_order_acquire:
    case memory_order_consume:
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      break;
    case memory_order_release:
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      break;
    case memory_order_acq_rel:
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
      break;
    case memory_order_seq_cst:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
      break;
    case memory_order_relaxed: break;
  };
}

__device__ inline void atomic_thread_fence_block(memory_order order) {
  switch(order) {
    case memory_order_acquire:
    case memory_order_consume:
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
      break;
    case memory_order_release:
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
      break;
    case memory_order_acq_rel:
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
      break;
    case memory_order_seq_cst:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
      break;
    case memory_order_relaxed: break;
  };
}

__device__ inline void atomic_thread_fence(
    memory_order order,
    thread_scope scope = thread_scope_system) {
  switch(scope) {
    case thread_scope_system: atomic_thread_fence_system(order); break;
    case thread_scope_device: atomic_thread_fence_device(order); break;
    case thread_scope_block: atomic_thread_fence_block(order); break;
    case thread_scope_wave:
    case thread_scope_thread: break;
  };
}

__device__ inline void waitcnt(thread_scope scope = thread_scope_system) {
  switch(scope) {
    case thread_scope_system:
      __builtin_amdgcn_s_waitcnt(0); // vmcnt(0) expcnt(0) lgkmcnt(0)
    case thread_scope_device:
      __builtin_amdgcn_s_waitcnt(0x3ff0); // vmcnt(0)
      break;
    case thread_scope_block:
      __builtin_amdgcn_s_waitcnt(0xc0ff); // lgkmcnt(0)
      break;
    case thread_scope_wave:
    case thread_scope_thread: break;
  };
}

template <thread_scope scope>
class barrier {
  public:
  using arrival_token = uint32_t;

  barrier() = delete;

  barrier(const barrier&)            = delete;
  barrier& operator=(const barrier&) = delete;

  barrier(uint32_t* ptr, uint32_t _expected)
      : count(ptr[0]), gen(ptr[1]), expected(_expected) {}

  __device__ inline arrival_token arrive(
      memory_order order = memory_order_seq_cst,
      uint32_t     n     = 1) {
    // get generation number for this barrier
    arrival_token g = gen.load(memory_order_relaxed);

    // increment counter
    const uint32_t old_count = count.fetch_add(n, order);

    // check if all threads have arrived
    if(old_count == expected - n) {
      // reset counter
      count.store(0, memory_order_relaxed);

      // Unlock other blocks
      gen.fetch_add(1, memory_order_relaxed);
    }

    return g;
  }

  __device__ inline void wait(arrival_token&& arrival,
                              memory_order order = memory_order_seq_cst) const {
    gen.wait(arrival, order);
  }

  __device__ inline void arrive_and_wait(
      memory_order order = memory_order_seq_cst) {
    wait(arrive(order), order);
  }

  __device__ inline bool is_last_to_arrive(uint32_t n = 1) {
    return (count.load(memory_order_relaxed) == expected - n);
  }

  private:
  atomic_ref<uint32_t, scope> count;
  atomic_ref<uint32_t, scope> gen;
  uint32_t                    expected = 0;
};

template <>
class barrier<thread_scope_block> {
  public:
  using arrival_token = uint32_t;

  barrier() = default;

  barrier(const barrier&)            = delete;
  barrier& operator=(const barrier&) = delete;

  __device__ inline arrival_token arrive(
      memory_order order = memory_order_seq_cst,
      uint32_t     n     = 1) {
    atomic_thread_fence_block(order);
    return 0;
  }

  __device__ inline void wait(arrival_token&& arrival,
                              memory_order order = memory_order_seq_cst) const {
    atomic_thread_fence_block(order);
    __builtin_amdgcn_s_barrier();
  }

  __device__ inline void arrive_and_wait(
      memory_order order = memory_order_seq_cst) {
    wait(arrive(order), order);
  }
};

} // namespace hip_ex

#ifndef HPL_USE_MOVE_DPP
#if defined(__GFX8__) || defined(__GFX9__)
#define HPL_USE_MOVE_DPP 1
#else
#define HPL_USE_MOVE_DPP 0
#endif
#endif

#if HPL_USE_MOVE_DPP
// DPP-based wavefront reduction maxloc
template <uint32_t WFSIZE>
__device__ inline void wavefront_maxloc(double& max, int& loc) {
  typedef union i64_b32 {
    double   f64;
    uint32_t u32[2];
  } f64_u32_t;

  f64_u32_t& r_max = reinterpret_cast<f64_u32_t&>(max);
  f64_u32_t  temp_max;
  int        temp_loc;

  temp_max.u32[0] = __hip_move_dpp(r_max.u32[0], 0x111, 0xf, 0xf, false);
  temp_max.u32[1] = __hip_move_dpp(r_max.u32[1], 0x111, 0xf, 0xf, false);
  temp_loc        = __hip_move_dpp(loc, 0x111, 0xf, 0xf, false);
  if(std::abs(temp_max.f64) > std::abs(r_max.f64)) {
    r_max.f64 = temp_max.f64;
    loc       = temp_loc;
  }

  temp_max.u32[0] = __hip_move_dpp(r_max.u32[0], 0x112, 0xf, 0xf, false);
  temp_max.u32[1] = __hip_move_dpp(r_max.u32[1], 0x112, 0xf, 0xf, false);
  temp_loc        = __hip_move_dpp(loc, 0x112, 0xf, 0xf, false);
  if(std::abs(temp_max.f64) > std::abs(r_max.f64)) {
    r_max.f64 = temp_max.f64;
    loc       = temp_loc;
  }

  temp_max.u32[0] = __hip_move_dpp(r_max.u32[0], 0x114, 0xf, 0xe, false);
  temp_max.u32[1] = __hip_move_dpp(r_max.u32[1], 0x114, 0xf, 0xe, false);
  temp_loc        = __hip_move_dpp(loc, 0x114, 0xf, 0xe, false);
  if(std::abs(temp_max.f64) > std::abs(r_max.f64)) {
    r_max.f64 = temp_max.f64;
    loc       = temp_loc;
  }

  temp_max.u32[0] = __hip_move_dpp(r_max.u32[0], 0x118, 0xf, 0xc, false);
  temp_max.u32[1] = __hip_move_dpp(r_max.u32[1], 0x118, 0xf, 0xc, false);
  temp_loc        = __hip_move_dpp(loc, 0x118, 0xf, 0xc, false);
  if(std::abs(temp_max.f64) > std::abs(r_max.f64)) {
    r_max.f64 = temp_max.f64;
    loc       = temp_loc;
  }

  temp_max.u32[0] = __hip_move_dpp(r_max.u32[0], 0x142, 0xa, 0xf, false);
  temp_max.u32[1] = __hip_move_dpp(r_max.u32[1], 0x142, 0xa, 0xf, false);
  temp_loc        = __hip_move_dpp(loc, 0x142, 0xa, 0xf, false);
  if(std::abs(temp_max.f64) > std::abs(r_max.f64)) {
    r_max.f64 = temp_max.f64;
    loc       = temp_loc;
  }

  if(WFSIZE > 32) {
    temp_max.u32[0] = __hip_move_dpp(r_max.u32[0], 0x143, 0xc, 0xf, false);
    temp_max.u32[1] = __hip_move_dpp(r_max.u32[1], 0x143, 0xc, 0xf, false);
    temp_loc        = __hip_move_dpp(loc, 0x143, 0xc, 0xf, false);
    if(std::abs(temp_max.f64) > std::abs(r_max.f64)) {
      r_max.f64 = temp_max.f64;
      loc       = temp_loc;
    }
  }
}

#else

template <uint32_t WFSIZE>
__device__ inline void wavefront_maxloc(double& max, int& loc) {

  for(int stride = WFSIZE / 2; stride > 0; stride >>= 1) {
    const int    temp_loc = __shfl_xor(loc, stride);
    const double temp_max = __shfl_xor(max, stride);

    if(std::abs(temp_max) > std::abs(max)) {
      loc = temp_loc;
      max = temp_max;
    }
  }
}

#endif

template <int BLOCKSIZE>
__device__ inline void block_maxloc(double& max,
                                    int&    loc,
                                    double* s_max,
                                    int*    s_loc) {
  const int t = threadIdx.x;

  s_max[t] = max;
  s_loc[t] = loc;
  __syncthreads();

  for(int active = BLOCKSIZE / 2; active > 0; active >>= 1) {
    if(t < active) {
      if(std::abs(s_max[t + active]) > std::abs(s_max[t])) {
        s_max[t] = s_max[t + active];
        s_loc[t] = s_loc[t + active];
      }
    }
    __syncthreads();
  }

  if(t < warpSize) {
    if(std::abs(s_max[t + warpSize]) > std::abs(s_max[t])) {
      max = s_max[t + warpSize];
      loc = s_loc[t + warpSize];
    } else {
      max = s_max[t];
      loc = s_loc[t];
    }
    wavefront_maxloc<WarpSize>(max, loc);
    s_max[t] = max;
    s_loc[t] = loc;
  }
  __syncthreads();
  max = s_max[warpSize - 1];
  loc = s_loc[warpSize - 1];
}

template <int BLOCKSIZE>
__device__ inline void atomic_block_maxloc(const int n,
                                           double*   arr,
                                           double*   s_max,
                                           int*      s_loc,
                                           double&   max,
                                           int&      loc) {
  max = 0.;
  loc = -1;
  for(int id = threadIdx.x; id < n; id += BLOCKSIZE) {
    hip_ex::atomic_ref<double, hip_ex::thread_scope_device> d_a(arr[id]);
    const double r_a = d_a.load(hip_ex::memory_order_relaxed);
    if(std::abs(r_a) > std::abs(max)) {
      max = r_a;
      loc = id;
    }
  }
  block_maxloc<BLOCKSIZE>(max, loc, s_max, s_loc);
}

#endif
