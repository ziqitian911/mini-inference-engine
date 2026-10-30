// Rigorous matmul benchmark.
//
// For each (matrix size, method) we:
//   1) verify correctness against the naive result (a fast-but-wrong kernel is useless),
//   2) run one warm-up pass (NOT timed) to pay cold-cache + first-touch page-fault costs,
//   3) time K runs and keep the MIN (the least-interfered run; average is polluted by noise),
//   4) sleep between configs to reduce progressive thermal throttling.
//
// Residual limitations we cannot fully control on a laptop: thermal throttling over a long
// run, and P-core vs E-core scheduling. So: trust a difference only if it is consistent
// across sizes AND clearly larger than the run-to-run jitter.

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
#include <functional>
#include <cmath>
#include <algorithm>

#include "tensor.h"
#include "gemm.h"
#include "gemm_cuda.h"

// Run fn once as warm-up (not timed), then K times; return the minimum seconds.
static double time_min(const std::function<void()> &fn, int K)
{
    fn(); // warm-up: absorb cold-cache / first-touch costs here, not in the measurement
    double best = 1e30;
    for (int r = 0; r < K; r++)
    {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

static double kernel_min(const std::function<float()> &fn, int K)
{
    fn();
    double best = 1e30;
    for (int r = 0; r < K; r++)
    {
        float result = fn();
        best = std::min((double)result, best);
    }

    return best / 1000; // scale it to second
}

// True if two tensors match element-wise within a small tolerance.
static bool same(const Tensor &a, const Tensor &b)
{
    if (a.size() != b.size())
        return false;
    for (int i = 0; i < a.size(); i++)
        if (std::abs(a[i] - b[i]) > 1e-3f)
            return false;
    return true;
}

int main()
{
    const int K = 5;         // timed runs per config (take the min)
    volatile float sink = 0; // keep results "used" so the optimizer can't delete the work

    std::cout << std::fixed << std::setprecision(4);

    for (int N : {512, 1024, 2048})
    {
        std::vector<float> vals(N * N, 1.0f);
        Tensor A({N, N}, vals), B({N, N}, vals);

        Tensor ref = matmul(A, B, MatmulMode::Naive); // reference answer for correctness
        Tensor out({N, N}, vals);                     // scratch output for the kernel-only CUDA timings
        double flops = 2.0 * N * N * N;               // FLOPs for an N*N*N matmul

        auto run = [&](const std::string &name, MatmulMode mode, int BS)
        {
            // 1) correctness: must equal the naive result, else the timing is meaningless
            Tensor C = matmul(A, B, mode, BS);
            bool ok = same(C, ref);

            // 2) timing: warm-up + K runs, keep the min
            double sec = time_min(
                [&]
                {
                    Tensor t = matmul(A, B, mode, BS);
                    sink = sink + t[0]; // touch the result
                },
                K);

            double gflops = flops / sec / 1e9;
            std::cout
                << "N=" << N
                << "  " << std::left << std::setw(15) << name << std::right
                << sec << " s   "
                << std::setw(8) << gflops << " GFLOPS   "
                << "correct=" << (ok ? "yes" : "NO!!")
                << "\n";

            std::this_thread::sleep_for(std::chrono::milliseconds(300)); // cool down a bit
        };

        auto run_kernel = [&](const std::string &name, const std::function<float()> &fn)
        {
            double sec = kernel_min(fn, K);
            double gflops = flops / sec / 1e9;

            std::cout
                << "N=" << N
                << "  " << std::left << std::setw(15) << name << std::right
                << sec << " s   "
                << std::setw(8) << gflops << " GFLOPS   "
                << "\n";

            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        };

        run("naive", MatmulMode::Naive, 0);
        run("reorder", MatmulMode::Reorder, 0);
        run("threaded", MatmulMode::Threaded, 0);
        run("threadedpool", MatmulMode::ThreadedPool, 0);
        run("cuda", MatmulMode::CUDA, 0);
        run_kernel(
            "cuda-kernel",
            [&]
            { return cuda_matmul(A.begin(), B.begin(), out.begin(), N, N, N); });
        run("blocked_cuda", MatmulMode::BlockedCUDA, 0);
        run_kernel(
            "blocked-kernel",
            [&]
            { return blocked_cuda_matmul(A.begin(), B.begin(), out.begin(), N, N, N); });
        run("blocked-32", MatmulMode::Blocked, 32);
        run("blocked-64", MatmulMode::Blocked, 64);
        run("blocked-128", MatmulMode::Blocked, 128);
        run("blocked-256", MatmulMode::Blocked, 256);
        std::cout << "\n";
    }

    return 0;
}
