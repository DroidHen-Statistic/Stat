import matplotlib.pyplot as plt
import math
import numpy as np
import random

def f(x):
    return x + math.sin(2 * math.pi * x)

seq = np.arange(0,10,0.01)
x = []
for i in range(25):
    print(len(seq))
    s = random.sample(list(seq), 10)
    seq = set(seq) - set(s)
    x = x + s
    y = [f(a) for a in s]
    plt.plot(s,y,'ob')
    plt.ylim(-1,11)
    plt.xlim(0,11)
    plt.title("sample = " + str((i+1) * 10))
    plt.savefig("E:\\公司\\科普向PPT\\素材\\" + str(i))
    # plt.show()
    # 
    # 



