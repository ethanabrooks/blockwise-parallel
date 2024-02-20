import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def worker(task_id, queue):
    result = (task_id) + 1
    queue.put(result)


if __name__ == "__main__":
    with mp.Manager() as manager:
        data_queue = manager.Queue()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker, i, data_queue) for i in range(10)]
            for future in futures:
                result = future.result()
                print(data_queue.get())
