#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// numa strategies
enum ggml_numa_strategy {
    GGML_NUMA_STRATEGY_DISABLED   = 0,
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
    GGML_NUMA_STRATEGY_ISOLATE    = 2,
    GGML_NUMA_STRATEGY_NUMACTL    = 3,
    GGML_NUMA_STRATEGY_MIRROR     = 4,
    GGML_NUMA_STRATEGY_COUNT
};

GGML_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node
GGML_API void    set_numa_thread_affinity(int thread_n);
GGML_API void    clear_numa_thread_affinity(void);

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};

GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

GGML_API struct ggml_threadpool * ggml_threadpool_new              (struct ggml_threadpool_params * params, void (*compute_thread) (void * data));
GGML_API void                     ggml_threadpool_free             (struct ggml_threadpool * threadpool);
GGML_API int                      ggml_threadpool_get_n_threads    (struct ggml_threadpool * threadpool);
GGML_API void                     ggml_threadpool_pause            (struct ggml_threadpool * threadpool);
GGML_API void                     ggml_threadpool_resume           (struct ggml_threadpool * threadpool);
GGML_API void                     ggml_threadpool_reset            (struct ggml_threadpool * threadpool, 
                                                                    struct ggml_cgraph * graph,
                                                                    void * backend_ctx);
GGML_API void                     ggml_threadpool_set_n_threads    (struct ggml_threadpool * threadpool, int n_threads);
GGML_API int                      ggml_threadpool_get_n_threads_max(struct ggml_threadpool * threadpool);
GGML_API void                     ggml_threadpool_run              (struct ggml_threadpool * threadpool, int thread_id);

GGML_API void                     ggml_graph_compute_kickoff       (struct ggml_threadpool * threadpool, int n_threads);
GGML_API void                     ggml_graph_compute_thread        (struct ggml_threadpool * threadpool, int thread_id);

GGML_API void                     ggml_barrier(struct ggml_threadpool * tp);

#ifdef __cplusplus
}
#endif
