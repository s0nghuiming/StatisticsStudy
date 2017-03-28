#第二章 感知机

##2.1 感知机模型
假设输入空间（特征空间）是$X\subseteq R^n$，输出空间是y={+1, -1}，输入x=X表示一个实例的特征向量，输出表示实例的类别。由输入空间到输出空间的函数$$f(x)=sign(w\centerdot x+b)$$称为感知机。其中$w\in R^n$是权值向量，$b \in R$是偏置，$w \centerdot x$是w和x的内积。sign是符号函数$$sign(x)=\begin{cases}+1,&x\geq 0\\-1,&x<0 \end{cases}$$
感知机是一种线性分类模型，判别模型。感知机的假设空间是定义在特征空间中的函数集合$$\{f|f(x)=w\centerdot x+b\}$$
感知机可以解释如下：线性方程$$w\centerdot x+b=0$$对应于特征空间$R^n$ 中的一个超平面。其中$w$是超平面的法向量，$b$是超平面的截距。这个超平面将特征空间分为2部分，也将特征空间里的点分为正、负两部分。所以，超平面也叫分离超平面。
##2.2 感知机学习策略
###线性可分与非
数据集T的线性可分性与非线性可分性。其中$$T=\left\{\left(x_1,y_1\right), \left(x_2,y_2\right),...\left(x_N,y_N\right)\right\}\\x_i\in{X}\subseteq{R^n}, y_i\in{Y=\left\{+1,-1\right\}}, i=1,2,...N$$
###感知机学习策略
定义损失函数
特征空间中任意选取一个点$x_0=\left(x_1^{(0)},x_2^{(0)},...x_N^{(0)}\right)$，点到超平面的距离$$l=\frac{{w}\centerdot{x}+b}{\parallel{w}\parallel}$$
对于一个误分类的点$\left(x_i,y_i\right)$，有$-y_i\left(w\centerdot{x_i}+b\right)\gt0$。因此，误分类点$x_i$到超平面S的距离为$$-\frac{1}{\parallel{w}\parallel}y_i\left(w\centerdot{x_i}+b\right)$$
对于给定的测试数据集
$$T=\left(\left(x_1,y_1\right),\left(x_2,y_2\right),...\left(x_N,y_N\right)\right)$$
其中$x_i\in{X}=R^n, y_i\in{Y}=\left\{+1,-1\right\}$。
感知机学习的损失函数定义为
$$L\left(w,b\right)=-\sum_{x_i\in{M}}y_i\left(w\centerdot{x_i}+b\right)$$
其中$M$是所有误分类点的集合。
##2.3 感知机学习算法
###2.3.1 算法的原型
$$\min_{w,b}L\left(w,b\right)$$
感知机学习算法是由误分类驱动的。 首先任意选取一个超平面$\left(w_0,b_0\right)$，然后用梯度下降法不断的极小化目标函数$L(w,b)$。对于误分类点的集合$M$，损失函数$L\left(w,b\right)$的梯度
$$\nabla_{w}L\left(w,b\right)=-\sum_{x_i\in{M}}{y_i}{x_i}\\\nabla_{b}L\left(w,b\right)=-\sum_{x_i\in{M}}{y_i}$$
随机选取一个误分类点$\left(x_i,y_i\right)$，对$w,b$进行更新：
$$w\leftarrow{w+\eta{y_i}{x_i}}\\b\leftarrow{b+\eta{y_i}}$$
式子中$\eta(0\lt\eta\leq1)$是步长，在统计学中叫做学习率。

**算法2.1 感知机学习的原始形式**
输入：训练数据集$T=\left\{\left(x_1,y_1\right), \left(x_2,y_2\right),...\left(x_N,y_N\right)\right\}\\x_i\in{X}\subseteq{R^n}, y_i\in{Y=\left\{+1,-1\right\}}, i=1,2,...N；学习率\eta(0\lt\eta\leq1)$
输出：$w,b；f(x)=sign(w\centerdot{x}+b)$
（1）选取初值：$w=w_0, b=b_0$
（2）在训练集$T$中选取数据$(x_i,y_i)$
（3）如果$y_i\left(wx_i+b\right)\leq{0}$
$$w\leftarrow{w+\eta{y_i}{x_i}}\\b\leftarrow{b+\eta{y_i}}$$
（4）转至（2），直到训练集中没有误分类点。
如下流程图。虽然画的丑了点儿，凑合看吧。
```flow
st=>start: Start
op=>operation: Update(w,b)
cond=>condition: IsMisFound?
en=>end: Over
st->cond
cond(no)->en
cond(yes)->op->cond
```
###2.3.2
 算法的收敛性
###2.3.3 算法的对偶形式
对偶形式的运算相对的快？不过其实这也是没什么卵用。那么多的乘法，CPU肯定吃不消，GPU？算了开始笔记。

##算法实现
