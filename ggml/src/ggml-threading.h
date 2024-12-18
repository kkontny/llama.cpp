#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

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

GGML_API struct ggml_threadpool * ggml_threadpool_new              (struct ggml_threadpool_params * params, void (*compute_function) (void * data));
GGML_API void                     ggml_threadpool_free             (struct ggml_threadpool * threadpool);
GGML_API int                      ggml_threadpool_get_n_threads    (struct ggml_threadpool * threadpool);
GGML_API void                     ggml_threadpool_pause            (struct ggml_threadpool * threadpool);
GGML_API void                     ggml_threadpool_resume           (struct ggml_threadpool * threadpool);
GGML_API void                     ggml_threadpool_reset            (struct ggml_threadpool * threadpool, struct ggml_cgraph * graph);
GGML_API void                     ggml_threadpool_set_n_threads    (struct ggml_threadpool * threadpool, int n_threads);
GGML_API int                      ggml_threadpool_get_n_threads_max(struct ggml_threadpool * threadpool);
GGML_API void                     ggml_threadpool_run              (struct ggml_threadpool * threadpool, int thread_id);

GGML_API void                     ggml_graph_compute_kickoff       (struct ggml_threadpool * threadpool, int n_threads);
GGML_API void                     ggml_graph_compute_thread        (void * data);

GGML_API void                     ggml_barrier(struct ggml_threadpool * tp);

#ifdef __cplusplus
}
#endif
