// ggml-ane.mm — Apple Neural Engine backend for ggml
// Uses maderix/ANE bridge for MIL compilation and ANE execution

#include "ggml-impl.h"
#include "ggml-ane.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ane_bridge.h"

#include <cstring>
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
    char buf[4096];
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
// MIL generation for fused QKV projection (3 conv1x1 from same input)
// ============================================================

static std::string gen_fused_qkv_mil(int D, int N, int Dq, int Dk, int Dv) {
    // Input: [1, N, 1, D] (ggml row-major)
    // Wq: [Dq, D, 1, 1], Wk: [Dk, D, 1, 1], Wv: [Dv, D, 1, 1]
    // Output: concat of Q[1,N,1,Dq], K[1,N,1,Dk], V[1,N,1,Dv]
    // Total output: [1, N, 1, Dq+Dk+Dv]
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
        "        tensor<int32, [4]> perm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,3,2,1])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x = transpose(perm=perm, x=x_raw)[name=string(\"xT\")];\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> q = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string(\"q\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> k = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=x)[name=string(\"k\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> v = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=x)[name=string(\"v\")];\n"
        "        int32 axis = const()[name=string(\"ax\"), val=int32(1)];\n"
        "        bool interleave = const()[name=string(\"il\"), val=bool(false)];\n"
        "        tensor<fp16, [1,%d,1,%d]> qkv = concat(axis=axis, interleave=interleave, values=(q, k, v))[name=string(\"qkv\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = transpose(perm=perm, x=qkv)[name=string(\"yT\")];\n"
        "    } -> (y);\n"
        "}\n",
        N, D,                // input [1, N, 1, D]
        D, N,                // transposed [1, D, 1, N]
        Dq, D, Dq, D,       // Wq
        Dk, D, Dk, D,       // Wk
        Dv, D, Dv, D,       // Wv
        Dq, N,               // Q output
        Dk, N,               // K output
        Dv, N,               // V output
        Dq+Dk+Dv, N,        // concat
        N, Dq+Dk+Dv);       // transposed output
    return std::string(buf);
}

// ============================================================
// fp32 <-> fp16 conversion
// ============================================================

static void fp32_to_fp16(const float * src, uint16_t * dst, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &src[i], 4);
        uint16_t sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        uint16_t frac = (bits >> 13) & 0x3FF;
        if (exponent <= 0) dst[i] = sign;
        else if (exponent >= 31) dst[i] = sign | 0x7C00;
        else dst[i] = sign | (exponent << 10) | frac;
    }
}

static void fp16_to_fp32(const uint16_t * src, float * dst, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t frac = h & 0x3FF;
        uint32_t bits;
        if (exponent == 0) bits = sign;
        else if (exponent == 31) bits = sign | 0x7F800000 | (frac << 13);
        else bits = sign | ((exponent - 15 + 127) << 23) | (frac << 13);
        memcpy(&dst[i], &bits, 4);
    }
}

// ============================================================
// Kernel cache
// ============================================================

struct ane_cache_key {
    int M, K, N;
    bool operator==(const ane_cache_key & o) const { return M == o.M && K == o.K && N == o.N; }
};
struct ane_cache_key_hash {
    size_t operator()(const ane_cache_key & k) const {
        return std::hash<int>()(k.M) ^ (std::hash<int>()(k.K) << 16) ^ (std::hash<int>()(k.N) << 32);
    }
};

struct ggml_backend_ane_context {
    std::string name = "ANE";
    bool bridge_ok = false;
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

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op != GGML_OP_MUL_MAT) continue;

        const struct ggml_tensor * src0 = node->src[0]; // weights [K, M]
        const struct ggml_tensor * src1 = node->src[1]; // input [K, N]

        int M = (int)src0->ne[1];
        int K = (int)src0->ne[0];
        int N = (int)src1->ne[1];

        if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) continue;
        if (M > 16384 || K > 16384 || N > 16384 || N < 1) continue;

        // Get or compile kernel
        ane_cache_key key = {M, K, N};
        ANEKernelHandle * kernel = nullptr;
        auto it = ctx->cache.find(key);
        if (it != ctx->cache.end()) {
            kernel = it->second;
        } else if (ctx->compile_count < 100) {
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
                ctx->cache[key] = kernel;
                ctx->compile_count++;
            }
        }

        if (!kernel) continue;

        // Convert input to fp16 (no transpose — MIL handles layout)
        std::vector<uint16_t> in_fp16(K * N);
        if (src1->type == GGML_TYPE_F32) {
            fp32_to_fp16((const float *)src1->data, in_fp16.data(), K * N);
        } else if (src1->type == GGML_TYPE_F16) {
            memcpy(in_fp16.data(), src1->data, K * N * 2);
        } else {
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
        }
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
        // ANE is efficient at N >= 64 (prefill), inefficient at N=1 (decode)
        // At N >= 128, ANE reaches 3.5+ TFLOPS — competitive with Metal
        // Let Metal/CPU handle decode (N < 64)
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
