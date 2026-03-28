// Test ggml ANE backend via dynamic loading
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("=== ggml ANE Backend Test ===\n");

    // Load all backends (including ANE)
    ggml_backend_load_all_from_path("build/bin");

    // Find ANE device
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

    printf("\nUsing: %s\n", ggml_backend_dev_name(ane_dev));

    // Init backend from device
    ggml_backend_t ane = ggml_backend_dev_init(ane_dev, NULL);
    if (!ane) {
        printf("FAILED: backend init returned NULL\n");
        return 1;
    }

    printf("Backend: %s\n", ggml_backend_name(ane));

    // Create simple MUL_MAT (N >= 64 for ANE)
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
    // Input: column c of each row = c+1 (so identity * input should give same)
    // ggml [K,N]: row s, col k = x[s*K + k]
    // We want output[s*M + m] = input[s*K + m] when weight=identity
    float * xd = (float *)x->data;
    for (int i = 0; i < K * N; i++) {
        int s = i / K;  // sequence index
        int k = i % K;  // channel index
        xd[i] = (float)(k + 1);  // each row is [1,2,3,...,K]
    }

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    printf("\nComputing MUL_MAT [%d,%d] x [%d,%d] on ANE...\n", M, K, K, N);
    ggml_backend_graph_compute(ane, graph);

    float * yd = (float *)y->data;
    // With identity weight and each row = [1,2,...,K]:
    // output should also have each row = [1,2,...,M]
    // ggml dst [M,N] row-major: row s = [s*M ... s*M+M-1]
    printf("Output row 0 [0..4]: %.2f %.2f %.2f %.2f %.2f\n",
           (double)yd[0], (double)yd[1], (double)yd[2], (double)yd[3], (double)yd[4]);
    printf("Expected row 0:      1.00 2.00 3.00 4.00 5.00\n");
    printf("Output row 1 [M..M+4]: %.2f %.2f %.2f %.2f %.2f\n",
           (double)yd[M], (double)yd[M+1], (double)yd[M+2], (double)yd[M+3], (double)yd[M+4]);

    int errors = 0;
    for (int s = 0; s < N; s++) {
        for (int m = 0; m < M; m++) {
            float expected = (float)(m + 1);
            float got = yd[s * M + m];
            if (fabsf(got - expected) > 1.0f) {
                if (errors < 5) printf("  Mismatch[s=%d,m=%d]: got %.2f expected %.2f\n", s, m, (double)got, (double)expected);
                errors++;
            }
        }
    }

    if (errors == 0) printf("\nPASSED!\n");
    else printf("\nFAILED: %d mismatches (fp16 precision may cause small diffs)\n", errors);

    ggml_free(ctx);
    ggml_backend_free(ane);
    return 0;
}
