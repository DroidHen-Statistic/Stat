import bisect
import time

def runtime(f):
    def inner(*args, **kwargs):
        t1 = time.time()
        f(*args, **kwargs)
        t2 = time.time()
        print("%s run time : %.fs" % (f.__name__, t2 - t1))
    return inner

@runtime
def f(n):
    a = 0
    for i in range(n):
        a += i
    print(a)

f(100000000)