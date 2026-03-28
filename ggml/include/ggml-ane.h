// ggml-ane.h — Apple Neural Engine backend for ggml
// Runs matrix multiplications on ANE via private AppleNeuralEngine.framework APIs
// Proof of concept: MUL_MAT only, expressed as conv1x1 in MIL

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_ANE_MAX_DEVICES 1

GGML_BACKEND_API ggml_backend_t ggml_backend_ane_init(void);

GGML_BACKEND_API bool ggml_backend_is_ane(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ane_reg(void);

#ifdef __cplusplus
}
#endif
