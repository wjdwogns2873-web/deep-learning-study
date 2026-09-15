#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    size_t maxSize_;

public:
    explicit ThreadSafeQueue(size_t maxSize = 2) : maxSize_(maxSize) {}

    // 큐에 데이터 삽입 (Producer)
    void push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);

        // 큐가 가득 차면 이전 최신 프레임을 버리고 새 프레임을 넣어 지연 방지
        if (queue_.size() >= maxSize_) {
            queue_.pop();
        }

        queue_.push(std::move(value));
        cond_.notify_one(); // 대기 중인 Consumer 스레드 깨우기
    }

    // 큐에서 데이터 꺼내기 (Consumer)
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);

        // 데이터가 올 때까지 대기
        cond_.wait(lock, [this] { return !queue_.empty(); });

        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};