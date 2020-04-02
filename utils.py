import time
import numpy as np


def show_progress(func, iterable, **kwargs):
    results = []
    times = []
    total = len(iterable)
    for i, item in enumerate(iterable):
        start = time.time()
        results.extend(func(item, **kwargs))
        times.append(time.time() - start)
        avg = np.mean(times)
        eta = avg * total - avg * (i + 1)
        print("Progress %d/%d for %s - ETA: %2fs" % (i + 1, total, item, eta), end="\r")
    return results