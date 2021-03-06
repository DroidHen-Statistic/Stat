import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
# from collections import defaultdict
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# print(head_path)
sys.path.append(head_path)
import config
from utils import *
# plt.switch_backend('agg')  # 服务器上跑一定要加这句，平时可以不要


data = np.array([1, 2, 3, 4, 3, 2, 6, 1, 3, 7, 5, 2, 4, 6])
data = data.reshape(-1, 2)
y = data[:, 1]
index = 0
for y_v in y:
    if y_v % 2 == 0:
        y_v = 1
    else:
        y_v = 0
    y[index] = y_v
    index += 1

######################################看 train_new.py，495行开始的标准使用################################


############################################### 简单生成样本 ###################################
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_regression
X, y, coef = make_regression(n_samples=1000, n_features=1, noise=10, coef=True)
# print(coef)
plt.scatter(X, y,  color='black')
plt.plot(X, X * coef, "b--",  linewidth=3)
plt.ylim((-100, 100))

# plt.xticks(())
# plt.yticks(())

plt.show()

# x = range(200)
# y =

# y = [pay_coins[int(len(pay_coins)/4.0 * 3)] for pay_coins in coins.values()]
# x = range(len(y))
# plt.plot(x, y,'o-')
# plt.xticks(x, coins.keys())
# plt.boxplot(list(coins.values()), labels = list(coins.keys()), sym = "")
# plt.gca().set_xlabel('uid')
# plt.gca().set_ylabel('coins')
# plt.show()


# fig = plt.figure()
plt.figure()
# ax = fig.add_subplot(1,1,1)
ax = plt.gca()
ax.plot(x, y, 'k--')
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                   rotation=45, fontsize='small')
ax.set_title('Demo Figure')
ax.set_xlabel('Time')
plt.show()
exit()


###################################### 3d绘图 ##################################
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.gcf()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax1=fig.add_subplot(221) # 增加一个子图，注意，fig要用add
# # ax1 = fig.add_subplot(111, projection="3d")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.scatter(data[:, 0], data[:, 1], y)
# # ax.
# # plt.legend()
# plt.show()

ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y) # 这个函数是生成x，y的点对，可以用简单的测试一下就知道
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
exit()


# 散点图，可以用y指定颜色
X1 = [1, 2, 3, 4, 5]
X2 = [3, 4, 5, 6, 7]
y = [0, 1, 0, 1, 2]
plt.scatter(X1, X2, marker='o', c=y)
figure_path = file_util.get_figure_path("slot", "machine_used")
file_name = os.path.join(figure_path, "123.png")
plt.savefig(file_name)
plt.show()
exit()


##################### 子图##################### 
# fig, ax = plt.subplots(2,2)
# print(fig, ax)

# ax = plt.gca()
# ax.plot(x, y)
# fig = plt.gcf()
# fig = plt.figure(2)
# fig.show()
# plt.show()
# print(ax)
s1 = plt.subplot(1, 2, 1)  # 把画布分成一行，两列，画在第一个位置
print(s1)

# print(type(ax))
s1.plot(x, y, 'r')

s2 = plt.subplot(2, 2, 2)  # 把画布分成两行，两列，画在第二个位置
s2 = plt.subplot(1, 1, 1)
s2.plot(x, y, 'b')
ax = plt.gca()
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5)) # 按0.5步长标示刻度
ax.xaxis.set_major_locator(plt.FixedLocator([2, 4, 6, 9]))  # 标示特定刻度
# ax.xaxis.set_major_locator(plt.FixedLocator(3,1))
# x = [2,4,6,9]
# ax.set_xticks(x)
# ax.xlim() #图像显示的范围
# ax.xaxis.set_major_locator(plt.LinearLocator(5)) #整个刻度线性分成5部分
# ax.xaxis.set_major_locator(plt.LogLocator(3, [1.0, 2.0])) # 按对数画刻度,第一个参数底数，第二个参数是倍数。比如这个例子在1,3,9,27...上标注刻度，也在2,6,12...上标注
# ax.xaxis.set_major_locator(plt.AutoLocator()) # 自动标注
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))   # 设置次要刻度

# ax.xaxis.set_ticklabels(['a','b','c','d']) # 设置刻度名称


# plt.FixedLocator([0, 2, 8, 9, 10])
print(s2, ax)

# s2 = plt.subplot(2,2,3) #把画布分成两行，两列，画在第三个位置
# s2.plot(x, y, 'b')

# s2 = plt.subplot(2,2,4) #把画布分成两行，两列，画在第四个位置
# s2.plot(x, y, 'b')
plt.show()


def test_plt():
    ax = plt.gca()
    ax.xlim = [0, 10]
    ax.ylim = [0, 10]
    ax.spine['top'] = plt.color("black")
    # ax.set
    ax.set_xlabel("test x")
    ax.set_ylabel("test y")
    ax.plot(x, y)
    plt.show()

# test_plt()

exit()

##################### 换x坐标，这几个一定要搭配起来用！#############################
ax = plt.gca()
xlim = [0, None] # 把显示范围固定
ax.set_xlim(xlim)
ax.xaxis.set_major_locator(plt.MultipleLocator(1)) # 按1步长标示刻度
ax.set_xticklabels([-1] + ["a", "b", "c"]) # 补一个-1，因为坐标原点强制不显示，不补就错位

exit()
##################### 换x坐标，这几个一定要搭配起来用！#############################



# 设置字体的样子，图片大小
gcf = plt.figure(figsize=(10, 4))
#gcf = plt.figure()
ax = plt.gca()
axisx = ax.xaxis
for label in axisx.get_ticklabels():
    label.set_color("red")
    label.set_rotation(45)  # 设置旋转字体
ax.set_title("title")
ax.set_xlabel("Lv group")
# xlim=(0, X_label[-1])
# ax.set_xlim(xlim)
y_expect = [1, 2, 3]
cr_X_label = [0, 1, 2]
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_ylabel("use counts")
ax.plot(y_expect, '--.', label="except")


# 加上标注，要改这个不能直接用
for pos in range(len(y_expect)):
    # plt.text(pos, y , total[pos] ,color='b',fontsize=2)
    if total[pos] > 0:
        # plt.text(pos, y[pos], "total: %s" % total[pos])
        # 在pos, y[pos]的位置显示pos+10，字体大小为7
        plt.text(pos, y_expect[pos], pos + 10, fontsize=7)


# ax.plot(y, '-.', label="real")
ax.set_xticklabels([-1, 0] + cr_X_label)
plt.annotate('total counts above curve', xy=(0, 0), xytext=(
    0.2, 0.2), arrowprops=dict(facecolor='blue', shrink=0.1))
gcf.savefig(file_name, dpi='160')  # 保存，分辨率


# plt.close(0) # 关闭第0个
plt.close('all')  # 关闭所有

exit()


x = np.linspace(0, 2 * math.pi, 100)
y = np.sin(x)

# x = [1,4,6,8,10]
x = [0, 1, 2, 3, 4]
y = np.array(x)
y[-1] = 10
print(x, y)
# ax = plt.gca()
# ax.xlim = (-1, 6)
# ax.plot(x, y)
# ax.xaxis.set_ticklabels([0, 'a','b','c','d','e']) # 设置刻度名称

xlim = (0, 6)
# xlim = (0,5)
s1 = plt.subplot(2, 2, 1)
s2 = plt.subplot(2, 2, 2)
s3 = plt.subplot(2, 1, 2)

# 画柱状图
s1.bar(x, y)
s1.set_xlim(xlim)
s1.xaxis

line = s2.plot(x)
print(line)
s2.set_xlim(xlim)
# 设置刻度名称，注意，最左边的刻度要刚好能画到图上，否则第一个刻度会被吃掉。
s2.xaxis.set_ticklabels(['a', 'b', 'c', 'd', 'e'])

s3.boxplot([x, y], sym=".")  # 线箱图

plt.show()
exit()

# 具体成员相关

axis = plt.gca().xaxis
plt.gca().plot([1, 2, 3], [4, 5, 6])

print(axis)
print(axis.get_ticklocs())
print(axis.get_ticklabels())

for label in axis.get_ticklabels():
    label.set_color("red")
    label.set_rotation(45)
    label.set_fontsize(16)
    # label.set_text("test")

for line in axis.get_ticklines():
    line.set_color("green")
    line.set_markersize(25)
    line.set_markeredgewidth(3)

txt = [x.get_text() for x in axis.get_ticklabels()]
print(txt)

# for t in txt:
#     print(t)
plt.show()
exit()


# 给fig添加对象
x = [1, 2, 3, 4]
fig = plt.figure()
ax1 = fig.add_subplot(221)  # 增加一个子图，注意，fig要用add
# ax2=fig.add_axes([1,1,1,1])
ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.3])  # 增加一个子图，四个参数不懂

for ax in fig.axes:  # 遍历axes，显示格子
    ax.grid(True)

"""注意为了让所创建的Line2D对象使用fig的坐标，我们将fig.TransFigure赋给Line2D对象的transform属性；
为了让Line2D对象知道它是在fig对象中，我们还设置其figure属性为fig；
最后还需要将创建的两个Line2D对象添加到fig.lines属性中去。
"""
from matplotlib.lines import Line2D
line1 = Line2D([0, 1], [0, 5], transform=fig.transFigure,
               figure=fig, color="r")
fig.lines.extend([line1])  # 整个画布上画像

np.random.seed(0)
n, bins, rects = ax1.hist(np.random.randn(1000), 50, facecolor="blue")  # 画直方图

plt.plot(x)
plt.show()
exit()


for idx, color in enumerate("rgbyck"):
    plt.subplot(320 + idx + 1, facecolor=color)
    # 如果希望在程序中调节的话，可以调用subplots_adjust函数，
    # 它有left, right, bottom, top, wspace,
    # hspace等几个关键字参数，这些参数的值都是0到1之间的小数，它们是以绘图区域的宽高为1进行正规化之后的坐标或者长度。
plt.subplots_adjust(bottom=0.1, top=0.9)  # 设置底边距等
x = [1, 2, 3, 4]
lines = plt.plot([1, 2, 3, 4])
print(lines)
for line in lines:
    print(plt.getp(line, 'color'))  # 取出参数，注意，是plt调用！
fig = plt.gcf()

fig2 = plt.figure()  # 一个新图
plt.plot(x)
plt.figure(1)  # 切换到第一个图
s1 = plt.subplot(3, 2, 1)
s1.plot(x)
ax.set_xscale("log", basex=2)  # 对数坐标轴
# fig2.subplot(222).plot(x)
# fig.patch.set_color("g") #设置背景颜色
plt.show()
exit()