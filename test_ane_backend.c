// Test ggml ANE backend via dynamic loading
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int test_identity_matmul(ggml_backend_t ane) {
    printf("\n--- Test 1: Identity matmul ---\n");
    int M = 256, K = 256, N = 64;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    // Identity weight
    float * wd = (float *)w->data;
    for (int i = 0; i < M * K; i++) {
        wd[i] = ((i / K) == (i % K)) ? 1.0f : 0.0f;
    }
    float * xd = (float *)x->data;
    for (int i = 0; i < K * N; i++) {
        xd[i] = (float)((i % K) + 1);
    }

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_graph_compute(ane, graph);

    float * yd = (float *)y->data;
    int errors = 0;
    for (int s = 0; s < N; s++) {
        for (int m = 0; m < M; m++) {
            float expected = (float)(m + 1);
            float got = yd[s * M + m];
            // fp16 roundtrip: max error for values 1-256 should be < 0.5
            if (fabsf(got - expected) > 0.5f) {
                if (errors < 5) printf("  Mismatch[s=%d,m=%d]: got %.4f expected %.4f\n", s, m, (double)got, (double)expected);
                errors++;
            }
        }
    }

    printf("Identity matmul: %s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
    ggml_free(ctx);
    return errors;
}

static int test_scaled_matmul(ggml_backend_t ane) {
    printf("\n--- Test 2: Scaled matmul (non-identity weights) ---\n");
    int M = 128, K = 128, N = 64;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    // Weight: 2x identity (output should be 2x input)
    float * wd = (float *)w->data;
    for (int i = 0; i < M * K; i++) {
        wd[i] = ((i / K) == (i % K)) ? 2.0f : 0.0f;
    }
    float * xd = (float *)x->data;
    for (int i = 0; i < K * N; i++) {
        xd[i] = (float)((i % K) + 1);
    }

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_graph_compute(ane, graph);

    float * yd = (float *)y->data;
    int errors = 0;
    for (int s = 0; s < N && s < 4; s++) {
        for (int m = 0; m < M; m++) {
            float expected = 2.0f * (float)(m + 1);
            float got = yd[s * M + m];
            if (fabsf(got - expected) > 1.0f) {
                if (errors < 5) printf("  Mismatch[s=%d,m=%d]: got %.4f expected %.4f\n", s, m, (double)got, (double)expected);
                errors++;
            }
        }
    }

    printf("Scaled matmul: %s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
    ggml_free(ctx);
    return errors;
}

static int test_different_weight_caching(ggml_backend_t ane) {
    printf("\n--- Test 3: Different weights same shape (cache key correctness) ---\n");
    int M = 128, K = 128, N = 64;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Two weight tensors, same shape, different values
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    // w1 = identity, w2 = 3x identity
    float * w1d = (float *)w1->data;
    float * w2d = (float *)w2->data;
    for (int i = 0; i < M * K; i++) {
        w1d[i] = ((i / K) == (i % K)) ? 1.0f : 0.0f;
        w2d[i] = ((i / K) == (i % K)) ? 3.0f : 0.0f;
    }
    float * xd = (float *)x->data;
    for (int i = 0; i < K * N; i++) xd[i] = 1.0f;

    struct ggml_tensor * y1 = ggml_mul_mat(ctx, w1, x);
    struct ggml_tensor * y2 = ggml_mul_mat(ctx, w2, x);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y1);
    ggml_build_forward_expand(graph, y2);
    ggml_backend_graph_compute(ane, graph);

    float * y1d = (float *)y1->data;
    float * y2d = (float *)y2->data;

    // y1 diagonal should be 1.0, y2 diagonal should be 3.0
    int errors = 0;
    for (int m = 0; m < M; m++) {
        if (fabsf(y1d[m] - 1.0f) > 0.5f) errors++;
        if (fabsf(y2d[m] - 3.0f) > 0.5f) errors++;
    }

    printf("Cache key correctness: %s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
    ggml_free(ctx);
    return errors;
}

int main(void) {
    printf("=== ggml ANE Backend Test ===\n");

    ggml_backend_load_all_from_path("build/bin");

    size_t n_devs = ggml_backend_dev_count();
    printf("Available devices: %zu\n", n_devs);

    ggml_backend_dev_t ane_dev = NULL;
    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        printf("  %s: %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        if (strstr(ggml_backend_dev_name(dev), "Neural") ||
            strstr(ggml_backend_dev_description(dev), "Neural")) {
            ane_dev = dev;
        }
    }

    if (!ane_dev) {
        printf("ANE device not found!\n");
        return 1;
    }

    ggml_backend_t ane = ggml_backend_dev_init(ane_dev, NULL);
    if (!ane) {
        printf("FAILED: backend init returned NULL\n");
        return 1;
    }

    int total_errors = 0;
    total_errors += test_identity_matmul(ane);
    total_errors += test_scaled_matmul(ane);
    total_errors += test_different_weight_caching(ane);

    printf("\n=== Summary: %s (%d total errors) ===\n",
           total_errors == 0 ? "ALL PASSED" : "SOME FAILED", total_errors);

    ggml_backend_free(ane);
    return total_errors > 0 ? 1 : 0;
}
