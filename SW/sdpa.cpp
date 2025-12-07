#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <sys/resource.h>
#include <sys/time.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cstdio> // para snprintf

#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include "timer.hpp"

constexpr int kBusWidthBits = 256;
constexpr int kElementBits = 32; // float32; usa 16 para half, etc.
constexpr int kPackets = kBusWidthBits / kElementBits;

double hw_ms = 0;
std::chrono::high_resolution_clock::time_point t_sync0, t_sync1, t_sync2, t_sync3;

// Función auxiliar para imprimir bytes en formato legible
static std::string human_bytes(size_t bytes) {
    char buf[64];
    const double kb = bytes / 1024.0;
    const double mb = kb / 1024.0;
    const double gb = mb / 1024.0;
    if (gb >= 1.0) { std::snprintf(buf, sizeof(buf), "%.3f GB", gb); return buf; }
    if (mb >= 1.0) { std::snprintf(buf, sizeof(buf), "%.3f MB", mb); return buf; }
    if (kb >= 1.0) { std::snprintf(buf, sizeof(buf), "%.3f KB", kb); return buf; }
    std::snprintf(buf, sizeof(buf), "%zu B", bytes); return buf;
}

static size_t round_up(size_t value, size_t multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

static void fill_matrix(std::vector<float>& mat, int rows, int cols, float base)
{
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            const size_t idx = static_cast<size_t>(r) * cols + c;
            mat[idx] = base + static_cast<float>((r + c) % 7) * 0.01f
                             + static_cast<float>(idx % 5) * 0.001f;
        }
    }
}

static std::vector<float> compute_reference(const std::vector<float>& Q,
                                            const std::vector<float>& K,
                                            const std::vector<float>& V,
                                            int l,
                                            int dk,
                                            int dv)
{
    std::vector<float> result(static_cast<size_t>(l) * dv, 0.0f);
    std::vector<float> scores(l);
    std::vector<float> softmax_row(l);
    const float inv_sqrt_dk = 1.0f / std::sqrt(static_cast<float>(dk));

    for (int q_idx = 0; q_idx < l; ++q_idx)
    {
        float max_score = -std::numeric_limits<float>::infinity();

        for (int k_idx = 0; k_idx < l; ++k_idx)
        {
            float score = 0.0f;
            for (int d = 0; d < dk; ++d)
            {
                score += Q[static_cast<size_t>(q_idx) * dk + d] *
                         K[static_cast<size_t>(k_idx) * dk + d];
            }
            score *= inv_sqrt_dk;
            scores[k_idx] = score;
            max_score = std::max(max_score, score);
        }

        float denom = 0.0f;
        for (int k_idx = 0; k_idx < l; ++k_idx)
        {
            const float val = std::exp(scores[k_idx] - max_score);
            softmax_row[k_idx] = val;
            denom += val;
        }
        const float inv_denom = 1.0f / denom;

        for (int k_idx = 0; k_idx < l; ++k_idx)
        {
            softmax_row[k_idx] *= inv_denom;
        }

        for (int v_col = 0; v_col < dv; ++v_col)
        {
            float acc = 0.0f;
            for (int k_idx = 0; k_idx < l; ++k_idx)
            {
                acc += softmax_row[k_idx] * V[static_cast<size_t>(k_idx) * dv + v_col];
            }
            result[static_cast<size_t>(q_idx) * dv + v_col] = acc;
        }
    }

    return result;
}

// Memoria pico del proceso (MB) — Linux
static double get_peak_rss_mb()
{
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return ru.ru_maxrss / 1024.0; // KB -> MB
    return -1.0;
}

int main(int argc, char** argv)
{
    INIT_PROFILER(cynq_profiler)

    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <l> <dk> <dv>\n";
        return EXIT_FAILURE;
    }

    const int l = std::stoi(argv[1]);
    const int dk = std::stoi(argv[2]);
    const int dv = std::stoi(argv[3]);

    if (l <= 0 || dk <= 0 || dv <= 0)
    {
        std::cerr << "All dimensions must be positive.\n";
        return EXIT_FAILURE;
    }

    const std::string binaryFile = "../HW/package.hw/kernels.xclbin";

    const size_t logical_q_elems = static_cast<size_t>(l) * dk;
    const size_t logical_k_elems = static_cast<size_t>(l) * dk;
    const size_t logical_v_elems = static_cast<size_t>(l) * dv;
    const size_t logical_a_elems = static_cast<size_t>(l) * dv;

    const size_t padded_q_elems = round_up(logical_q_elems, kPackets);
    const size_t padded_k_elems = round_up(logical_k_elems, kPackets);
    const size_t padded_v_elems = round_up(logical_v_elems, kPackets);
    const size_t padded_a_elems = round_up(logical_a_elems, kPackets);

    std::vector<float> host_q(padded_q_elems, 0.0f);
    std::vector<float> host_k(padded_k_elems, 0.0f);
    std::vector<float> host_v(padded_v_elems, 0.0f);
    std::vector<float> host_a(padded_a_elems, std::numeric_limits<float>::quiet_NaN());

    fill_matrix(host_q, l, dk, 0.05f);
    fill_matrix(host_k, l, dk, 0.11f);
    fill_matrix(host_v, l, dv, 0.17f);

    // === Tiempo SW (golden) ===
    auto sw_t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> golden = compute_reference(host_q, host_k, host_v, l, dk, dv);
    auto sw_t1 = std::chrono::high_resolution_clock::now();
    const double sw_ms =
        std::chrono::duration<double, std::milli>(sw_t1 - sw_t0).count();
    const double sw_peak_mb = get_peak_rss_mb();


    auto bytes_of = [](size_t elems){ return elems * sizeof(float); };

    const size_t sw_qb = bytes_of((size_t)l * dk);
    const size_t sw_kb = bytes_of((size_t)l * dk);
    const size_t sw_vb = bytes_of((size_t)l * dv);
    const size_t sw_ab = bytes_of((size_t)l * dv);
    // temporales del golden:
    const size_t sw_scores_b     = bytes_of((size_t)l);
    const size_t sw_softmax_b    = bytes_of((size_t)l);
    const size_t sw_result_b     = sw_ab; // l*dv
    const size_t sw_total_countable =
    sw_qb + sw_kb + sw_vb + sw_ab + sw_scores_b + sw_softmax_b + sw_result_b;


    GET_PROFILE_INSTANCE(setup_time, cynq_profiler);
    setup_time->reset();

    const int device_index = 0;
    std::cout << "Opening device " << device_index << '\n';
    auto device = xrt::device(device_index);
    std::cout << "Loading xclbin " << binaryFile << '\n';
    auto uuid = device.load_xclbin(binaryFile);
    auto sdpa = xrt::kernel(device, uuid, "sdpa");
    setup_time->tick();


    try
    {
        std::cout << "Allocating buffers\n";
        auto bo_q = xrt::bo(device, sizeof(float)*padded_q_elems, sdpa.group_id(0));
        auto bo_k = xrt::bo(device, sizeof(float)*padded_k_elems, sdpa.group_id(1));
        auto bo_v = xrt::bo(device, sizeof(float)*padded_v_elems, sdpa.group_id(2));
        auto bo_a = xrt::bo(device, sizeof(float)*padded_a_elems, sdpa.group_id(6));

        // === Memoria global (DDR/HBM) usada por el kernel ===
        // Cada xrt::bo se aloja en la memoria global de la FPGA (DDR o HBM).
        const size_t bytes_q = sizeof(float) * padded_q_elems;
        const size_t bytes_k = sizeof(float) * padded_k_elems;
        const size_t bytes_v = sizeof(float) * padded_v_elems;
        const size_t bytes_a = sizeof(float) * padded_a_elems;
        const size_t bytes_total = bytes_q + bytes_k + bytes_v + bytes_a;

        auto q_map = bo_q.map<float*>();
        auto k_map = bo_k.map<float*>();
        auto v_map = bo_v.map<float*>();
        auto a_map = bo_a.map<float*>();

        std::copy(host_q.begin(), host_q.end(), q_map);
        std::copy(host_k.begin(), host_k.end(), k_map);
        std::copy(host_v.begin(), host_v.end(), v_map);
        std::fill(a_map, a_map + padded_a_elems, 0.0f);

        // --- Overhead timing: Host -> Device (H2D) ---
        t_sync0 = std::chrono::high_resolution_clock::now();
        bo_q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        t_sync1 = std::chrono::high_resolution_clock::now();

        START_PROFILE(kernel_execution, cynq_profiler, 10)
        // === Tiempo HW (solo ejecución del kernel) ===
        auto run = sdpa(bo_q, bo_k, bo_v,
                        static_cast<uint32_t>(l),
                        static_cast<uint32_t>(dk),
                        static_cast<uint32_t>(dv),
                        bo_a);
        run.wait();
        t_sync2 = std::chrono::high_resolution_clock::now();
        hw_ms = std::chrono::duration<double, std::milli>(t_sync2 - t_sync1).count();
        END_PROFILE(kernel_execution);

        // --- Overhead timing: Device -> Host ---
        bo_a.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        t_sync3 = std::chrono::high_resolution_clock::now();

        std::copy(a_map, a_map + padded_a_elems, host_a.begin());

        const float tolerance = 1e-2f;
        float max_diff = 0.0f;
        size_t max_idx = 0;
        size_t mismatches = 0;

        for (size_t idx = 0; idx < logical_a_elems; ++idx)
        {
            const float diff = std::fabs(host_a[idx] - golden[idx]);
            if (diff > max_diff)
            {
                max_diff = diff;
                max_idx = idx;
            }
            if (diff > tolerance)
            {
                ++mismatches;
            }
        }

        std::cout << "Max abs diff: " << max_diff << " at idx " << max_idx << '\n';
        std::cout << "Mismatches (> " << tolerance << "): " << mismatches << '\n';

        // === Software reference metrics ===
        std::cout << "\n=== Software reference metrics (Time/Memory) ===\n";
        std::cout << "Software ref time: " << sw_ms << " ms\n";
        std::cout << "\n=== Software (CPU) memory ===\n";
        std::cout << "Q: " << human_bytes(sw_qb) << "\n";
        std::cout << "K: " << human_bytes(sw_kb) << "\n";
        std::cout << "V: " << human_bytes(sw_vb) << "\n";
        std::cout << "A: " << human_bytes(sw_ab) << "\n";
        std::cout << "scores: "    << human_bytes(sw_scores_b)  << "\n";
        std::cout << "softmax_row: "<< human_bytes(sw_softmax_b) << "\n";
        std::cout << "result: "     << human_bytes(sw_result_b)  << "\n";
        std::cout << "TOTAL(countable): " << human_bytes(sw_total_countable) << "\n";
        if (sw_peak_mb >= 0.0) std::cout << "Software ref peak RSS: " << sw_peak_mb << " MB\n";
        else                   std::cout << "Software ref peak RSS: N/A\n";


        // === Kernel metrics (Time/Memory) ===
        std::cout << "\n=== Kernel metrics (Time/Memory) ===\n";
        std::cout << "Kernel time: " << hw_ms << " ms\n";
        std::cout << "Q buffer: " << human_bytes(bytes_q) << " (" << padded_q_elems << " floats)\n";
        std::cout << "K buffer: " << human_bytes(bytes_k) << " (" << padded_k_elems << " floats)\n";
        std::cout << "V buffer: " << human_bytes(bytes_v) << " (" << padded_v_elems << " floats)\n";
        std::cout << "A buffer: " << human_bytes(bytes_a) << " (" << padded_a_elems << " floats)\n";
        std::cout << "TOTAL   : " << human_bytes(bytes_total) << "\n";
        std::cout << "======================================\n";

        // === Host–Device Overhead breakdown ===
        const double h2d_ms   = std::chrono::duration<double, std::milli>(t_sync1 - t_sync0).count();
        const double d2h_ms   = std::chrono::duration<double, std::milli>(t_sync3 - t_sync2).count();
        const double e2e_ms   = h2d_ms + hw_ms + d2h_ms;
        const double xfer_ms  = h2d_ms + d2h_ms;
        const double xfer_pct = (e2e_ms > 0.0) ? (100.0 * xfer_ms / e2e_ms) : 0.0;

        std::cout << "\n=== Host–Device Overhead ===\n";
        std::cout << "Host -> Device: " << h2d_ms << " ms\n";
        std::cout << "Kernel exec        : " << hw_ms << " ms\n";
        std::cout << "Device -> Host: " << d2h_ms << " ms\n";
        std::cout << "End-to-end: " << e2e_ms << " ms\n";
        std::cout << "Transfer overhead %: " << xfer_pct << " %\n";
        std::cout << "============================\n";

        std::cout << cynq_profiler << std::endl;

        const bool pass = mismatches == 0;
        std::cout << (pass ? "TEST PASSED" : "TEST FAILED") << '\n';
        return pass ? EXIT_SUCCESS : EXIT_FAILURE;

    }
    catch (const std::exception& e)
    {
        std::cerr << "BO alloc exception: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

}
