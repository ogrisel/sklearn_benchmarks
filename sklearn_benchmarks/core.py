import time


class BenchmarkTimer:
    @staticmethod
    def run(func, *args):
        start = time.perf_counter()
        res = func(*args)
        end = time.perf_counter()
        return (res, end - start)
