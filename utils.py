import time
import hashlib
import numpy as np


def show_progress(func, iterable, **kwargs):
    """
    Wraps a function to privide and expected time of arrival
    func (callable): the function you want to compute
    iterable (iterable): an iterable to loop through

    Use **kwargs to provide additional inputs to the function
    """
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


def get_image_id(array):
    """
    Returns the sha1 hex representation of a numpy array
    see: https://gist.github.com/epignatelli/75cf84b1534a1e817ea36004dfd52e6a
    for performance tests
    """
    return hashlib.sha1(array).hexdigest()
