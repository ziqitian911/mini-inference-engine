#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <functional>

class ThreadPool
{
private:
    std::vector<std::thread> threads;        // Vector to store worker threads
    std::queue<std::function<void()>> tasks; // Queue for tasks
    std::condition_variable cv;              // conditional variable
    std::mutex mtx;                          // Mutex to synchronize access to shared data
    bool stop;

    void worker_loop();

public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    void enqueue(std::function<void()> task);
};
