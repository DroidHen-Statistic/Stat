## Stat

运行环境python3

需要scipy，numpy，matplotlib

windows下建议直接安装Anaconda



### 目录结构

1. item_used	item_get		return	level_left 文件夹

   分别存放处理对应log的脚本

2. figures文件夹

   保存脚本运行过程中绘制的各种图片

3. papers文件夹

   保存了一些参考文献

4. MysqlConnection.py

   Python的Mysql接口

5. utils.py

   一些工具函数

6. config.py

   配置参数和常量设置


#### return

return文件夹下两个脚本：

1. data_process.py 主要用于数据的预处理和可视化，内有多个相对来说比较独立的函数，具体函数功能如下：

```python
calculateTotalReturn(date_start, date_end):
	"""
	汇总每天总的留存数据，并以百分比的形式保存在新表中
	
	Arguments:
		date_start {int} -- 要统计的开始日期
		date_end {int} -- 要统计的结束日期
	"""
calculateChannelTotalReturn(date_start, date_end):
	"""
	统计不同channel_id的留存数据，并以百分比的形式保存在新表中
	
	Arguments:
		date_start {int} -- 要统计的开始日期
		date_end {int} -- 要统计的结束日期
	"""
dateReturn(date_start, date_end, channels = [-2]):
	"""
	某天注册的用户留存情况
	
	获取某一天注册的用户留存数据，并以天数为横坐标，留存率为纵坐标，每天做出一张图
	
	Arguments:
		date_start {int} -- 要统计的开始日期
		date_end {int} -- 要统计的结束日期
	
	Keyword Arguments:
		channels {list} -- 要统计的channel_id,以列表形式给出，-2 为统计总和 (default: {[-2]})
	"""
dayReturn(days):
	"""
	n-day留存情况
	
	获取不同日期下的同一个n-day 留存（2<=n<=30）,并以日期为横坐标，留存率为纵坐标，每一个n作出一张图
	
	Arguments:
		days {list} -- 要统计的n日留存，以列表方式给出
	"""
dauAnddnu():
	"""
	绘制每日活跃用户、新增用户以及两者差值（老用户）的曲线图
	
	"""
dauAnddnu_Bar():
	"""
	dau和dnu的直方图
	"""
dnuOfChannelId(channels = [-2]):
	"""[summary]
	
	不同channel_id的dnu人数柱状图
	
	Keyword Arguments:
		channels {list} -- 要统计的channel_id 列表，-2为全部 (default: {[-2]})
	"""
dnuOfChannelID_Percent(channels = [-1]):
	"""
	dnu不同channel占总数的比例曲线图
	
	Keyword Arguments:
		channels {list} -- 要统计的channel，以列表形式给出 (default: {[-1]})
	"""
```

2. data_fit.py 主要用来日留存用户的拟合建模，只有一个函数：

```python
fitDateReturn(game_id, channels = [-2]):
	"""
	拟合每天新用户的30日留存曲线

	采用scipy 的 curve_fit，原理是最小二乘

	利用10折交叉验证，选取了error最小的一次作为最终模型
	
	Arguments:
		game_id {int} -- 游戏id
	
	Keyword Arguments:
		channels {list} -- 要拟合的渠道信息，-2为总和 (default: {[-2]})
	
	Returns:
		best_popt {list} -- 拟合得到的参数值
		best_pcov {list} -- 系数的协方差值，后续暂时没有用到
	"""
```

curve_fit函数是scipy中的一个模型拟合函数，采用

关于交叉验证，其作用是进行模型选择、防止数据的过拟合等，主要的形式是k-折交叉验证，而其中10-折交叉验证又是最常用的。10-折交叉验证是将数据集分为10份，每次训练用其中的9份作为训练集，剩下的一份作为验证集，如此重复10次，将10次得到的模型的性能指标做一个平均，来作为最终的模型评价。这样做的好处在于可以一定程度上表征出模型是否有过拟合现象的出现。

交叉验证的主要功能是进行模型选择，即存在多个候选模型的情况下，可以利用交叉验证来选取评价指标最好的那一个作为最后的模型。但是，这里我们采用的模型实际只有一个指数模型，所以，交叉验证的意义可能不在模型选择上。我的想法是，利用交叉验证，来一定程度上去除一些异常数据，使得模型能够更准确一些，同时提供一个模型的评价指标。

#### level_left

level_left下只有一个脚本文件 data_process.py 主要用于数据的处理和可视化，主要函数功能如下：

```python
level7DayLeft(levels):
	"""
	各个等级在不同日期的7日留存数
	
	按天统计各个等级7日留存，并以日期为横坐标，留存为纵坐标，每个等级画出一张图
	
	Arguments:
		levels {list} -- 要统计的等级，以列表形式给出
	"""
date7DayLeft(dates):
	"""
	每日的各等级7日留存统计
	
	统计同一天的各个等级的7日留存数据，并以等级为横坐标，留存为纵坐标，每天画出一张图
	
	Arguments:
		dates {list} -- 要统计的日期，以列表形式给出
	"""
	
levelTotal(start,end):
	"""
	画出每个等级的7日流失人数总和分布
	
	获取每个等级的7日流失用户总数，并以等级为横坐标，人数为纵坐标画图
	
	Arguments:
		start {int} -- 要统计的起始等级
		end {int} -- 要统计的结束等级
	"""
relativeLevelLeft(start,end):
	"""
	对应等级范围的相对流失曲线
	
	相对流失定义为该等级的流失人数除以前五个等级流失人数的平均数
	
	Arguments:
		start {int} -- 起始等级
		end {int} -- 结束等级
	"""
```

#### item_used

item_used 主要用来进行item的相关性分析。

##### 相关算法

相关性分析的主要算法受启发于推荐算法中的经典：协同滤波。但是我们目前的目标不需要进行用户级别的个性推荐，所以这里我只是借鉴了协同滤波中第一步算法：计算物品相似度。

算法的基本思想很简单，构建用户---道具矩阵，每行为每个用户，每列为每个道具，矩阵的元素为用户使用每个道具的次数（或者购买次数、评价等能够反映用户对该物品感兴趣程度或者满意度的度量值），这样，每个物品可以表示为一个列向量，计算物品之间的相似度转化为计算向量之间的相似度。但是，为了保证准确性，我们在计算时，只拿出两个物品都使用过的用户分量。

向量相似度的计算有多种方式，这里程序采用的是pearson相关性系数。

向量x和向量y的pearson相关性系数定义为：

$$sim(x,y) = \frac{(x-\bar x)\cdot (y - \bar y)}{\sqrt{(x-\bar x)\cdot (x-\bar x)^T}\sqrt{(y-\bar y)\cdot (y-\bar y)^T}}$$

式中的$$\bar x$$和$$\bar y$$是x和y的均值向量，所以pearson相关性系数其实也就是x和y的协方差除以二者的标准差之积。

其中协方差表示了二者的变化趋势的相似度，而除以标准差则是做了一个归一化。

item_used下含两个脚本文件：

1. data_process.py 主要用来处理原始log，并计算user—item table，log格式为

   ```php
   class ItemUsedFormat
   {
       // <item_used_2>uid stage_id item_id count data_version is_first
       const UID_POS = 0;
       const STAGE_ID_POS = 1; // 关卡外为-1
       const ITEM_ID_POS = 2;
       const COUNT_POS = 3;
       const DATA_VERSION_POS = 4;
       const FIRST_POS = 5;
       //14 20 1002 1 1 0
   }
   ```

   主要函数功能如下：

```python
readLog(day_dir):
	"""
	读取一天的log文件，统计每个玩家各个道具的使用数量
	
	Arguments:
		day_dir {string} -- 该日的log文件夹路径
	
	Returns:
		dict -- 二维dict，{uid_1:{item_id_1:使用次数,...,item_id_n:使用次数},...,uid_n:{item_id_1:使用次数,...,item_id_n:使用次数}}
	"""
calculateUserItemTable(years = -1, months = -1, days = -1):
	"""
	计算user_item表，元素为每个user使用各个item的次数
	
	Arguments:
		date_start {int} -- 统计起始日期
		date_end {int} -- 统计终止日期
		game_id {int} -- 游戏id
	"""
```

2. calculate_sim.py 用来计算item之间的相关度，主要函数只有一个：

```python
sim(game_id):
	"""
	计算item之间的相似度，这里采用的是pearson相关性系数
	
	Arguments:
		game_id {int} -- game_id
	"""
```

#### item_get

item_get和item_used的功能基本相同，只是读取的log不一样， log格式：

```PHP
class ItemGetFormat
{
    // <item_get_2>uid stage_id item_id count data_version is_first via
    const UID_POS = 0;
    const STAGE_ID_POS = 1; // 关卡外为-1
    const ITEM_ID_POS = 2;
    const COUNT_POS = 3;
    const DATA_VERSION_POS = 4;
    const FIRST_POS = 5;
    const VIA_POS = 6; // 方式
    // 14 20 1021 1 1 0 3
}
```


