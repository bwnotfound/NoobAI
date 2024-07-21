import math

from threading import Thread
from queue import Queue
from tqdm import tqdm
import time


class _RunThread(Thread):
    def __init__(self, run_func_list, result_queue: Queue, thread_id, step_queue=None):
        super().__init__()
        self.run_func_list = run_func_list
        self.result_queue = result_queue
        self.thread_id = thread_id
        self.step_queue = step_queue

    def run(self):
        result_list = []
        for run_func in self.run_func_list:
            result_list.append(run_func())
            if self.step_queue is not None:
                self.step_queue.put(None)
        self.result_queue.put((self.thread_id, result_list))

# A simple multithread work gather
def do_work(
    run_func_list,
    num_workers,
    show_progress=True,
    block_or_step="block",
    sleep_time=None,
):
    if not isinstance(run_func_list, (list, tuple)):
        run_func_list = [run_func_list]
    block_size = math.ceil(len(run_func_list) / num_workers)
    result_queue = Queue()
    result_list_dict = {}
    if show_progress:
        step_queue = Queue()
    t_bar = tqdm(total=len(run_func_list), colour="green", ncols=80)
    splited_func_list_list = []
    if block_or_step == "block":
        for start_idx in range(0, len(run_func_list), block_size):
            splited_func_list_list.append(
                run_func_list[start_idx : start_idx + block_size]
            )
    else:
        for offset in range(num_workers):
            if offset >= len(run_func_list):
                break
            splited_func_list_list.append(run_func_list[offset::num_workers])
    for thread_idx, splited_func_list in enumerate(splited_func_list_list):
        _RunThread(
            splited_func_list, result_queue, thread_idx, step_queue=step_queue
        ).start()
    thread_num = len(splited_func_list_list)
    while thread_num > 0:
        while not step_queue.empty():
            step_queue.get()
            t_bar.update()
        while not result_queue.empty():
            thread_result = result_queue.get()
            result_list_dict[thread_result[0]] = thread_result[1]
            thread_num -= 1
        if sleep_time is not None:
            time.sleep(sleep_time)
    while not step_queue.empty():
        step_queue.get()
        t_bar.update()
    # t_bar.close()
    result_list_list = []
    for i in range(len(splited_func_list_list)):
        result_list_list.append(result_list_dict[i])
    result_list = []
    if block_or_step == "block":
        for i in range(len(splited_func_list_list)):
            result_list.extend(result_list_dict[i])
    else:
        for i in range(len(run_func_list)):
            step, offset = i // num_workers, i % num_workers
            result_list.append(result_list_list[offset][step])

    return result_list
