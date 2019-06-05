//
// Created by wei on 3/23/18.
//

#pragma once
#include <atomic>
#include <thread>
#include <mutex>
#include <array>
#include <list>
#include <functional>
#include <condition_variable>

namespace surfelwarp {
	
	template<int NumThreads = 2>
	class ThreadPool final {
	private:
		//An array of valid threads, created upon
		//the contruction of ThreadPool class
		std::array<std::thread, NumThreads> m_threads;
		
		//The task queue
		std::list<std::function<void(void)>> m_job_queue;
		
		//Atomic accessed variables
		std::atomic_int m_jobs_left;
		std::atomic_bool m_bailout;
	
		//The mutex and condvar for access to the job queue
		//All the worker threads might block on this condvar
		std::mutex m_queue_mutex;
		std::condition_variable m_job_available_condvar;
		
		//The mutex and condition for waiting
		//The main threads will block on this condvar
		std::mutex m_thread_wait_mutex;
		std::condition_variable m_thread_wait_condvar;

	public:
		//Construct the threadpool will create threads
		//and invoke their running of threadFunc, which
		//fetch jobs from job queue and block if empty
		ThreadPool()
			: m_jobs_left(0),
			  m_bailout(false)
		{
			//Construct the threads
			for(auto i = 0; i < NumThreads; i++) {
				m_threads[i] = std::thread([this]{this->threadFunc();});
			}
		}
		
		//The destructor will wait all job to be finished
		//and destroy the threads
		~ThreadPool() {
			//Let everyone finish
			WaitAll();
			
			//Do not accept new tasks
			m_bailout = true;
			m_job_available_condvar.notify_all();
			
			//Join the threads
			for(std::thread& thread : m_threads) {
				if(thread.joinable()) {
					thread.join();
				}
			}
		}
		
		//No copy/assign
		ThreadPool(const ThreadPool&) = delete;
		ThreadPool& operator=(const ThreadPool&) = delete;
		
		//Push a new job
		void PushJob(std::function<void(void)> job) {
			std::lock_guard<std::mutex> guard(m_queue_mutex);
			m_job_queue.emplace_back(job);
			++m_jobs_left;
			m_job_available_condvar.notify_one();
		}
		
		//Wait for all job in the job queue to be finished
		void WaitAll() {
			if(m_jobs_left > 0) {
				std::unique_lock<std::mutex> lock(m_thread_wait_mutex);
				//block the main threads on condition variable
				m_thread_wait_condvar.wait(lock, [this]()->bool{
					return (this->m_jobs_left == 0);
				});
				lock.unlock();
			}
		}
	
	private:
		//Obtain the next job
		std::function<void(void)> nextJob() {
			std::function<void(void)> job;
			std::unique_lock<std::mutex> job_lock(m_queue_mutex);
			
			//Wait for a job if there is no one available
			m_job_available_condvar.wait(job_lock, [this]()->bool{
				return (m_job_queue.size() > 0 || m_bailout);
			});
			
			//There is one job available
			if(!m_bailout) {
				job = m_job_queue.front();
				m_job_queue.pop_front();
			}
			else {
				job = []{}; // An empty job
				++m_jobs_left;
			}
			
			//The job to be executed
			return job;

			//The job_lock will be released after
			//the scope of this function
		}
		
		//The processing function for threads
		void threadFunc() {
			while(!m_bailout) {
				nextJob()();
				--m_jobs_left;
				m_thread_wait_condvar.notify_one();
			}
		}
	};
}