import os
import time
from contextlib import contextmanager

@contextmanager
def timer(query_index, module, perf_folder):
    save_file = r"{}/{}.csv".format(perf_folder, module)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    save_perf(save_file, query_index, elapsed)

def save_perf(file, query_index, elapsed):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("Index, run_time\n")
    with open(file, "a") as metric_file:
        metric_file.write(
            str(query_index) + "," + str(elapsed) + "\n")


if __name__ == '__main__':
    for i in range(10):
        with timer(i, "test_module", "."):
            time.sleep(0.1)