// ggml-ane.mm — Apple Neural Engine backend for ggml
// Uses maderix/ANE bridge for MIL compilation and ANE execution

#include "ggml-impl.h"
#include "ggml-ane.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ane_bridge.h"

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================
// MIL generation for conv1x1 (= matmul)
// ============================================================

static std::string gen_matmul_mil(int M, int K, int N) {
    // Input: ggml row-major [K,N] = N*K fp16 values
    // Interpret as [1, N, 1, K] then transpose to [1, K, 1, N] for conv
    // After conv [1, M, 1, N], transpose back to [1, N, 1, M] for ggml output
    char buf[8192];
    snprintf(buf, sizeof(buf),
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x_raw) {\n"
        "        tensor<int32, [4]> perm_in = const()[name=string(\"pi\"), val=tensor<int32, [4]>([0,3,2,1])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x = transpose(perm=perm_in, x=x_raw)[name=string(\"xT\")];\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), "
        "offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y_raw = conv(dilations=dl,groups=gr,pad=pd,"
        "pad_type=pt,strides=st,weight=W,x=x)[name=string(\"y\")];\n"
        "        tensor<int32, [4]> perm_out = const()[name=string(\"po\"), val=tensor<int32, [4]>([0,3,2,1])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = transpose(perm=perm_out, x=y_raw)[name=string(\"yT\")];\n"
        "    } -> (y);\n"
        "}\n",
        N, K,       // input: [1, N, 1, K] (ggml row-major)
        K, N,       // after transpose: [1, K, 1, N]
        M, K, M, K, // weight
        M, N,       // conv output: [1, M, 1, N]
        N, M);      // after transpose: [1, N, 1, M] (ggml row-major)
    return std::string(buf);
}

// ============================================================
// fp32 <-> fp16 conversion
// ============================================================

static void fp32_to_fp16(const float * src, uint16_t * dst, int n) {
    for (int i = 0; i < n; i++) {
        // Use _Float16 for correct rounding (round-to-nearest-even)
        // and proper denormal handling. Available on all Apple clang targets.
        _Float16 h = (_Float16)src[i];
        memcpy(&dst[i], &h, 2);
    }
}

static void fp16_to_fp32(const uint16_t * src, float * dst, int n) {
    for (int i = 0; i < n; i++) {
        _Float16 h;
        memcpy(&h, &src[i], 2);
        dst[i] = (float)h;
    }
}

// ============================================================
// Kernel cache
// ============================================================

struct ane_cache_key {
    int M, K, N;
    const void * weight_data;  // identifies which weight tensor (stable within a ggml graph)
    bool operator==(const ane_cache_key & o) const {
        return M == o.M && K == o.K && N == o.N && weight_data == o.weight_data;
    }
};
struct ane_cache_key_hash {
    size_t operator()(const ane_cache_key & k) const {
        return std::hash<int>()(k.M) ^ (std::hash<int>()(k.K) << 16)
             ^ (std::hash<int>()(k.N) << 32) ^ std::hash<const void *>()(k.weight_data);
    }
};

struct ggml_backend_ane_context {
    std::string name = "ANE";
    bool bridge_ok = false;
    std::mutex cache_mutex;
    std::unordered_map<ane_cache_key, ANEKernelHandle *, ane_cache_key_hash> cache;
    int compile_count = 0;
};

// ============================================================
// ggml backend interface
// ============================================================

static const char * ggml_backend_ane_get_name(ggml_backend_t backend) {
    return "ANE";
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_ane_context *>(backend->context);
    for (auto & kv : ctx->cache) {
        ane_bridge_free(kv.second);
    }
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_ane_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_ane_context *>(backend->context);
    int n_computed = 0;
    int n_attempted = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op != GGML_OP_MUL_MAT) continue;

        const struct ggml_tensor * src0 = node->src[0]; // weights [K, M]
        const struct ggml_tensor * src1 = node->src[1]; // input [K, N]

        int M = (int)src0->ne[1];
        int K = (int)src0->ne[0];
        int N = (int)src1->ne[1];

        if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) continue;
        // Match supports_op: M,K,N >= 64 and <= 16384
        if (M < 64 || K < 64 || N < 64 || M > 16384 || K > 16384 || N > 16384) continue;

        n_attempted++;

        // Get or compile kernel (thread-safe cache access)
        ane_cache_key key = {M, K, N, src0->data};
        ANEKernelHandle * kernel = nullptr;
        {
            std::lock_guard<std::mutex> lock(ctx->cache_mutex);
            auto it = ctx->cache.find(key);
            if (it != ctx->cache.end()) {
                kernel = it->second;
            }
        }
        if (!kernel && ctx->compile_count < 4096) {
            // Dequantize weights to fp16
            std::vector<uint16_t> w_fp16(M * K);
            if (src0->type == GGML_TYPE_F32) {
                fp32_to_fp16((const float *)src0->data, w_fp16.data(), M * K);
            } else if (src0->type == GGML_TYPE_F16) {
                memcpy(w_fp16.data(), src0->data, M * K * 2);
            } else {
                std::vector<float> w_f32(M * K);
                ggml_get_type_traits(src0->type)->to_float(src0->data, w_f32.data(), M * K);
                fp32_to_fp16(w_f32.data(), w_fp16.data(), M * K);
            }

            // Build weight blob — exact format from maderix build_blob()
            size_t w_bytes = M * K * 2;
            size_t blob_size = 128 + w_bytes;
            std::vector<uint8_t> blob(blob_size, 0);
            blob[0] = 1; blob[4] = 2;
            blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE;
            blob[68] = 1;
            uint32_t ws = (uint32_t)w_bytes;
            uint32_t data_off = 128;
            memcpy(&blob[72], &ws, 4);
            memcpy(&blob[80], &data_off, 4);
            memcpy(&blob[128], w_fp16.data(), w_bytes);

            std::string mil = gen_matmul_mil(M, K, N);
            size_t in_size = K * N * 2;
            size_t out_size = M * N * 2;

            kernel = ane_bridge_compile(mil.c_str(), mil.size(),
                                         blob.data(), blob_size,
                                         1, &in_size, 1, &out_size);
            if (kernel) {
                std::lock_guard<std::mutex> lock(ctx->cache_mutex);
                // Double-check: another thread may have compiled the same key
                auto it2 = ctx->cache.find(key);
                if (it2 != ctx->cache.end()) {
                    // Lost the race — free our duplicate and use theirs
                    ane_bridge_free(kernel);
                    kernel = it2->second;
                } else {
                    ctx->cache[key] = kernel;
                    ctx->compile_count++;
                }
            }
        }

        if (!kernel) {
            GGML_LOG_WARN("ANE: no kernel for MUL_MAT [%d,%d,%d]\n", M, K, N);
            continue;
        }

        // Convert input to fp16 (no transpose — MIL handles layout)
        std::vector<uint16_t> in_fp16(K * N);
        if (src1->type == GGML_TYPE_F32) {
            fp32_to_fp16((const float *)src1->data, in_fp16.data(), K * N);
        } else if (src1->type == GGML_TYPE_F16) {
            memcpy(in_fp16.data(), src1->data, K * N * 2);
        } else {
            GGML_LOG_WARN("ANE: unsupported src1 type %d for MUL_MAT\n", src1->type);
            continue;
        }

        ane_bridge_write_input(kernel, 0, (const uint8_t *)in_fp16.data(), K * N * 2);
        bool ok = ane_bridge_eval(kernel);

        if (ok) {
            std::vector<uint16_t> out_fp16(M * N);
            ane_bridge_read_output(kernel, 0, (uint8_t *)out_fp16.data(), M * N * 2);

            if (node->type == GGML_TYPE_F32) {
                fp16_to_fp32(out_fp16.data(), (float *)node->data, M * N);
            }
            n_computed++;
        } else {
            GGML_LOG_ERROR("ANE: eval failed for MUL_MAT [%d,%d,%d]\n", M, K, N);
        }
    }

    if (n_attempted > 0 && n_computed == 0) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_ane_i = {
    /* .get_name                = */ ggml_backend_ane_get_name,
    /* .free                    = */ ggml_backend_ane_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ane_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// ============================================================
// Device
// ============================================================

static const char * ggml_backend_ane_device_get_name(ggml_backend_dev_t dev) { return "Apple Neural Engine"; GGML_UNUSED(dev); }
static const char * ggml_backend_ane_device_get_description(ggml_backend_dev_t dev) { return "ANE via private API"; GGML_UNUSED(dev); }
static enum ggml_backend_dev_type ggml_backend_ane_device_get_type(ggml_backend_dev_t dev) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; GGML_UNUSED(dev); }

static void ggml_backend_ane_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) { *free = 0; *total = 0; GGML_UNUSED(dev); }

static void ggml_backend_ane_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_ane_device_get_name(dev);
    props->description = ggml_backend_ane_device_get_description(dev);
    props->type = ggml_backend_ane_device_get_type(dev);
    props->memory_free = 0; props->memory_total = 0;
    memset(&props->caps, 0, sizeof(props->caps));
    GGML_UNUSED(dev);
}

static ggml_backend_t ggml_backend_ane_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_ane_init(); GGML_UNUSED(dev); GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_ane_device_get_buffer_type(ggml_backend_dev_t dev) { return ggml_backend_cpu_buffer_type(); GGML_UNUSED(dev); }

static bool ggml_backend_ane_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (op->op == GGML_OP_MUL_MAT) {
        int64_t M = op->src[0]->ne[1], K = op->src[0]->ne[0], N = op->src[1]->ne[1];
        // src1 (input) must be F32 or F16 (the compute path skips other types)
        enum ggml_type t1 = op->src[1]->type;
        if (t1 != GGML_TYPE_F32 && t1 != GGML_TYPE_F16) return false;
        // ANE is efficient at N >= 64 (prefill), inefficient at N=1 (decode)
        return M >= 64 && K >= 64 && N >= 64 &&
               op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 &&
               op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1;
    }
    return false; GGML_UNUSED(dev);
}

static bool ggml_backend_ane_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_cpu_buffer_type(); GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_ane_device_i = {
    /* .get_name             = */ ggml_backend_ane_device_get_name,
    /* .get_description      = */ ggml_backend_ane_device_get_description,
    /* .get_memory           = */ ggml_backend_ane_device_get_memory,
    /* .get_type             = */ ggml_backend_ane_device_get_type,
    /* .get_props            = */ ggml_backend_ane_device_get_props,
    /* .init_backend         = */ ggml_backend_ane_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_ane_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_ane_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ane_device_supports_buft,
    /* .offload_op           = */ ggml_backend_ane_device_supports_op,  // offload all supported ops
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ============================================================
// Registration
// ============================================================

static const char * ggml_backend_ane_reg_get_name(ggml_backend_reg_t reg) { return "ANE"; GGML_UNUSED(reg); }
static size_t ggml_backend_ane_reg_get_device_count(ggml_backend_reg_t reg) { return 1; GGML_UNUSED(reg); }

static ggml_backend_dev_t ggml_backend_ane_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    static struct ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_ane_device_i,
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    dev.reg = ggml_backend_ane_reg();
    return &dev;
    GGML_UNUSED(reg); GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_ane_reg_i = {
    /* .get_name         = */ ggml_backend_ane_reg_get_name,
    /* .get_device_count = */ ggml_backend_ane_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ane_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_ane_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_ane_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

ggml_backend_t ggml_backend_ane_init(void) {
    if (ane_bridge_init() != 0) {
        GGML_LOG_ERROR("ANE: failed to init bridge\n");
        return nullptr;
    }

    auto * ctx = new ggml_backend_ane_context();
    ctx->bridge_ok = true;

    auto * backend = new ggml_backend {
        /* .guid    = */ {},
        /* .iface   = */ ggml_backend_ane_i,
        /* .device  = */ ggml_backend_ane_reg_get_device(nullptr, 0),
        /* .context = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_ane(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_ane_get_name;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ane_reg)
