#include "ggml-threading.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <new>

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#else

#include <pthread.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Per-thread state
struct alignas(std::hardware_destructive_interference_size) ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    bool cpumask[GGML_MAX_N_THREADS];
    int  last_graph;
    bool pending;
#endif
    ggml_threadpool * threadpool;
    int ith;
};

// Threadpool def
struct alignas(std::hardware_destructive_interference_size) ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;

    // synchronization primitives
    std::atomic_int n_graph;       // incremented when there is work to be done (i.e each graph)
    alignas(std::hardware_destructive_interference_size) std::atomic_int n_barrier;
    alignas(std::hardware_destructive_interference_size) std::atomic_int n_barrier_passed;
    std::atomic_int current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    std::atomic_bool stop;         // Used for stopping the threadpool altogether
    std::atomic_bool pause;        // Used for pausing the threadpool or individual threads
    std::atomic_bool abort;        // Used for aborting processing of a graph

    ggml_compute_state * workers;   // per thread state
    void (*compute_function) (void * data);  // function to run by each worker thread
    int               n_threads_max; // number of threads in the pool
    std::atomic_int   n_threads_cur; // number of threads used in the current graph

    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#else
inline void ggml_thread_cpu_relax(void) {;}
#endif

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        std::memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        std::memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

// check if thread is active
static bool ggml_graph_compute_thread_active(ggml_compute_state * state) {
    ggml_threadpool * threadpool = state->threadpool;
    int n_threads = std::atomic_load_explicit(&threadpool->n_threads_cur, std::memory_order_relaxed);
    return (state->ith < n_threads);
}

// check if thread is ready to proceed (exit from polling or sleeping)
static bool ggml_graph_compute_thread_ready(ggml_compute_state * state) {
    ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int new_graph = std::atomic_load_explicit(&threadpool->n_graph, std::memory_order_relaxed);
    if (new_graph != state->last_graph) {
        state->pending    = ggml_graph_compute_thread_active(state);
        state->last_graph = new_graph;
    }

    return state->pending;
}

// sync thread state after polling
static void ggml_graph_compute_thread_sync(ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    std::atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    std::atomic_thread_fence(std::memory_order_seq_cst);
    #endif
    GGML_UNUSED(state);
}

static bool ggml_graph_compute_poll_for_work(ggml_compute_state * state) {
    ggml_threadpool * threadpool = state->threadpool;

    // Skip polling for unused threads
    if (!ggml_graph_compute_thread_active(state)) {
        return state->pending;
    }

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static bool ggml_graph_compute_check_for_work(ggml_compute_state * state) {
    ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

#ifndef GGML_USE_OPENMP

// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    ggml_compute_state * state = (ggml_compute_state *) data;
    ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    GGML_ASSERT(threadpool->compute_function);

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            threadpool->compute_function(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
void ggml_graph_compute_kickoff(ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    GGML_PRINT_DEBUG("threadpool: n_threads_cur %d n_threads %d\n", threadpool->n_threads_cur, n_threads);

    // Update the number of active threads
    std::atomic_store_explicit(&threadpool->n_threads_cur, n_threads, std::memory_order_relaxed);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    std::atomic_fetch_add_explicit(&threadpool->n_graph, 1, std::memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif

ggml_threadpool * ggml_threadpool_new(
    ggml_threadpool_params * tpp,
    void (*compute_function) (void * data)) {

    ggml_threadpool * threadpool = new ggml_threadpool;
    {
        threadpool->cgraph           = nullptr;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = false;
        threadpool->workers          = nullptr;
        threadpool->compute_function = compute_function;
        threadpool->n_threads_max    = tpp->n_threads;
        threadpool->n_threads_cur    = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(ggml_compute_state) * tpp->n_threads;
    ggml_compute_state * workers = new ggml_compute_state[tpp->n_threads];

    std::memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifndef GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

void ggml_threadpool_free(ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads_max;

#ifndef GGML_USE_OPENMP
    ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    delete[] threadpool->workers;
    delete threadpool;
}

void ggml_threadpool_pause(ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_set_n_threads(ggml_threadpool * threadpool, int n_threads) {
    GGML_ASSERT(n_threads <= threadpool->n_threads_max);
    std::atomic_store_explicit(&threadpool->n_threads_cur, n_threads, std::memory_order_relaxed);
}

int ggml_threadpool_get_n_threads_max(ggml_threadpool * threadpool) {
    return threadpool->n_threads_max;
}

void ggml_threadpool_run(ggml_threadpool * threadpool, int thread_id) {
    threadpool->compute_function(&threadpool->workers[thread_id]);
}

void ggml_threadpool_reset(ggml_threadpool * threadpool, ggml_cgraph * cgraph) {
    threadpool->cgraph           = cgraph;
    threadpool->current_chunk    = 0;
    threadpool->abort            = false;
    threadpool->ec               = GGML_STATUS_SUCCESS;
}

void ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * threadpool    = state->threadpool;

    const struct ggml_cgraph * cgraph = threadpool->cgraph;
    const struct ggml_cplan  * cplan  = threadpool->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ atomic_load_explicit(&threadpool->n_threads_cur, std::memory_order_relaxed),
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ threadpool,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes && !threadpool->abort; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        threadpool->compute_function(&params, node);

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            threadpool->abort = true;
            threadpool->ec    = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->threadpool);
    }

    return;
}

void ggml_barrier(ggml_threadpool * threadpool) {
    int n_threads = std::atomic_load_explicit(&threadpool->n_threads_cur, std::memory_order_relaxed);
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = std::atomic_load_explicit(&threadpool->n_barrier_passed, std::memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = std::atomic_fetch_add_explicit(&threadpool->n_barrier, 1, std::memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        std::atomic_store_explicit(&threadpool->n_barrier, 0, std::memory_order_relaxed);

        // exit barrier (fill seq-cst fence)
        std::atomic_fetch_add_explicit(&threadpool->n_barrier_passed, 1, std::memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (std::atomic_load_explicit(&threadpool->n_barrier_passed, std::memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    std::atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, std::memory_order_seq_cst);
    #else
    std::atomic_thread_fence(std::memory_order_seq_cst);
    #endif
#endif
}
