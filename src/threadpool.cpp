#include "threadpool.h"

void ThreadPool::worker_loop()
{
    while (true)
    {
        std::function<void()> task;
        {
            // Locking the queue so that data can be shared safely
            std::unique_lock<std::mutex> lock(this->mtx);

            // Waiting until there is a task to execute or the pool is stopped
            this->cv.wait(
                lock,
                [this]() -> bool
                {
                    return !this->tasks.empty() || this->stop;
                });

            // exit the thread in case the pool is stopped and there are no tasks
            if (stop && this->tasks.empty())
            {
                return;
            }

            // Get the next task from the queue
            task = std::move(this->tasks.front());
            this->tasks.pop();
        }

        task();
    }
}

ThreadPool::ThreadPool(size_t num_threads) : stop(false)
{
    // Creating worker threads
    for (size_t t = 0; t < num_threads; t++)
    {
        this->threads.emplace_back(
            [this]()
            {
                this->worker_loop();
            });
    }
}

/**
 * @brief Destructor to stop the thread pool
 */
ThreadPool::~ThreadPool()
{
    {
        // Lock the queue to update the stop flag safely
        std::unique_lock<std::mutex> lock(this->mtx);
        this->stop = true;
    }

    // Notify all threads
    this->cv.notify_all();

    // Joining all worker threads to ensure
    // they have completed their tasks
    for (auto &thread : this->threads)
    {
        thread.join();
    }
}

/**
 * @brief Enqueue task for execution by the thread pool
 */
void ThreadPool::enqueue(std::function<void()> task)
{
    {
        std::unique_lock<std::mutex> lock(this->mtx);
        this->tasks.emplace(std::move(task));
    }
    this->cv.notify_one();
}
