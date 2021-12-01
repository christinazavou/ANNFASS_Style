import multiprocessing as mp
import time


def run_function_in_parallel(func, n_processes, data, **kwargs):
    """
    :param func the function to be called by multiple parallel processes
    :param n_processes how many processes
    :param data a list of the data to be chunked and passed in multiple calls of func
    :param args extra parameters to be passed in func
    e.g. if func is defined as func(d, p1, p2) then args is (p1, p2)
         if func is defined as func(d, p1=0, p2=0) then args can be (p1, p2) or {p1:0, p2:0}
    """
    step = int(round(len(data) / float(n_processes)))
    processes = []
    for process_id in range(n_processes):
        start_idx = process_id * step
        if process_id == n_processes - 1:
            end_idx = len(data)
        else:
            end_idx = (process_id + 1) * step
        data_chunk = data[start_idx:end_idx]
        if len(data_chunk) == 0:
            continue
        t = mp.Process(target=func, args=(data_chunk, process_id), kwargs=kwargs)
        processes.append(t)
        t.start()

    # Wait for processes to end
    for oneProc in processes:
        oneProc.join()


def log_process_time(process_id, start_time):
    elapsed_time = time.time() - start_time
    print("Terminating process {process_id}. Time passed: {hours:d}:{minutes:d}:{seconds:d}"
          .format(process_id=process_id,
                  hours=int((elapsed_time / 60 ** 2) % (60 ** 2)),
                  minutes=int((elapsed_time / 60) % (60)),
                  seconds=int(elapsed_time % 60)))
