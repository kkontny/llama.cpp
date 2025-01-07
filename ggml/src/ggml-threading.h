#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

GGML_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params * params);
GGML_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
GGML_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
GGML_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
GGML_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);
GGML_API void                          ggml_threadpool_reset         (struct ggml_threadpool * threadpool, struct ggml_cgraph * cgraph);
GGML_API void                          ggml_graph_compute_kickoff    (struct ggml_threadpool * threadpool, int n_threads);
GGML_API void                          ggml_threadpool_set_craph     (struct ggml_threadpool * threadpool, struct ggml_cgraph * cgraph);

#ifdef __cplusplus
}
#endif
