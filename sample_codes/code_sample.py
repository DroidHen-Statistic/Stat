import numpy as np
from collections import defaultdict


# 简单生成样本
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_regression
X, y, coef =make_regression(n_samples=1000, n_features=1,noise=10, coef=True)
# print(coef)
plt.scatter(X, y,  color='black')
plt.plot(X, X*coef, "b--",  linewidth=3)
plt.ylim((-100, 100))

# plt.xticks(())
# plt.yticks(())

plt.show()
exit()

# defaultdict的工厂函数
class A:
    def __init__(self):
         self._l = 3
    def fc(self):
         return np.zeros(self._l)
    def test(self):
         b = defaultdict(self.fc)
         # print(b)
         return b
    def test2(self)     :
        self.test()

a = A()
#a.fc()
b = a.test()
print(b[1])
a. test2()
exit()


class _point():
    def __init__(self,x,y):
        self.x = x
        self.y =y


        # 持久化
import pickle
shoplistfile = "shoplist.data"
#print(type(movie_file_after_read))
#print(movie_file_after_read)

shoplist = ['apple', 'mango', 'carrot']
shoplist = np.zeros((3,5))
shoplist = _point(3, 4)
with open(shoplistfile, 'wb') as f:
    pickle.dump(shoplist, f)

np.save("shoplist.npy", shoplist)

k = np.load("shoplist.npy")
with open("shoplist.data","rb") as f:
    k2 = pickle.load(f)


print(type(k))
print(type(k2))


# for循环初始化
a = dict(
    (movie_id, fan_count) for movie_id, fan_count in movie_2_fans_count.items() if fan_count > min_support

    )

#defaultdict接受工厂参数，设置default没用
a = defaultdict(lambda : 20)
print(a[2])

# set的操作
a = t | s          # t 和 s的并集  
b = t & s          # t 和 s的交集  
#c = t – s        # 求差集（项在t中，但不在s中）  
d = t ^ s          # 对称差集（项在t或s中，但不会同时出现在二者中）


# 滚动，移位
x = np.arange(10)
np.roll(x, 2)
np.roll(x, -2)


# csv reader
import csv
csv_file = os.path.join( os.path.dirname(os.path.abspath(__file__)), "data"+ os.path.sep + "CCPP" + os.path.sep + "Folds5x2_pp.csv")
with open(csv_file, "r") as f:
    reader = csv.reader(f)
    # reader = csv.reader(f, delimiter=" ")
    for row, line in reader:
        print (line)
        break


# 分类训练样板
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

estimator = KNeighborsClassifier()
estimator = SVC()
#estimator = DecisionTreeClassifier()
#estimator = RandomForestClassifier(random_state=14)
# estimator = ExtraTreesClassifier()
estimator = AdaBoostClassifier()
# estimator = GradientBoostingClassifier()
# print(data_2.shape)
# print(label_2.shape)
# estimator.fit(data_2,label_2)
# score = estimator.score(data_2_total[900:-1], label[900:-1])
# print(score)
score = np.mean(cross_val_score(estimator, data_2_total,
                                label, scoring="accuracy", cv=10))
print(score)

#创建等比数列
#假如，我们想要改变基数，不让它以10为底数，我们可以改变base参数，将其设置为2试试。
# base默认是10
# 下列含义 从2^0=1开始，到2^9=512结束，生成10个等比数
a = np.logspace(0,9,10,base=2)

# 奇巧淫技
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
np.arange(0, 10)[:, np.newaxis] # 把每行的元素再套一层，变成10*1的样子
np.arange(0, 10).reshape(-1,1) # 和上面一样的


# 简单流水线
clf = linear_model.Ridge()
from sklearn.pipeline import Pipeline
pip = Pipeline([('clf', clf)])
# pip.set_params(steps = [('clf2', clf)]) # 这个可以改变流程
# pip.named_steps['clf2'] = clf # 这个没用，不知道为啥
pip.fit(X_test, y_test)
pip.score(X_test, y_test, "neg_mean_absolute_error") # score的类型分类和回归是不一样的，这里要用回归的


# 画图
from matplotlib import pyplot as plt
x = np.linspace(0, 10, 1000)
y = np.sin(x)
z = np.cos(x**2)

# matplotlib.rcParams["savefig.dpi"]

plt.figure(figsize=((10, 8)))

plt.plot(x,z,"b--",label="$cos(x^2)$") #"--"表示虚线
print(plt.xlim())

# plt.set_xscale("log")
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("PyPlot First Example")
plt.ylim([-1.2, 1.2])
# plt.xlim(plt.xlim()[::-1])
plt.plot(x,y,label="$sin(x)$",color="red",linewidth=4) # $表示斜体

# plt.xticks(()) # 滤掉x轴和y轴
# plt.yticks(())

plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 5, 0.1)
line, = plt.plot(x, x*x) # plot返回一个列表，通过line,获取其第一个元素
# 调用Line2D对象的set_*方法设置属性值
line.set_antialiased(False)



exit()