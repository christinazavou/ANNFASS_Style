import multiprocessing as mp


# Create processes and run function in parallel
def runParallelFunc(func, nProc, data, argDict=None):
    # Create new processes
    step = int(round(len(data) / float(nProc)))
    processes = []
    for i in range(nProc):
        startIdx = i * step
        if (nProc - 1) > i:
            endIdx = (i + 1) * step
        else:
            endIdx = len(data)
        if argDict == None:
            t = mp.Process(target=func, args=(data[startIdx:endIdx], i))
        elif isinstance(argDict, dict):
            t = mp.Process(target=func, args=(data[startIdx:endIdx], i, argDict))
        else:
            print("TypeError: argDict is not of type dict() (type(argDict) == {_type:s}".format(_type=type(argDict)))
            exit(-1)
        processes.append(t)
        t.start()

    # Wait for processes to end
    for oneProc in processes:
        oneProc.join()
