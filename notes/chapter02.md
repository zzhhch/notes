# 模型评估与选择

## 2.1 经验误差与过拟合

- **错误率和精度**：$m$个样本中有$a$个样本分类错误，错误率$E=a/ m$，精度 = 1 - 错误率
- **误差**：	
  - 训练误差（经验误差）：学习器在训练集上的误差
  - 泛化误差：在新样本上的误差

- **过拟合与欠拟合**：过拟合由于学习器能力太强，将不一般的特性都学到了，欠拟合由于学习能力不足造成的

## 2.2 评估方法

- **测试集和测试误差**：用测试集来测试学习器对新样本的判别能力，用测试误差作为泛化误差的近似值（测试集也是从独立同分布的样本实例中选取的，应尽量与训练集互斥）
- 我们只有包含$m$个样例的数据集$D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)\}$,既要训练也要测试，则应对$D$进行适当的处理，从而产生训练集$S$和测试集$T$

### 2.2.1 留出法

- **留出法(hold-out)**：直接将数据集$D$划分为两个互斥的集合，其中一个作为训练集$S$,另一个作为测试集$T$

  为保证数据分布的一致性，从采样的角度，保留分类比例的采样方式通常是分成采样

### 2.2.2 交叉验证法

- **交叉验证法(cross validation)**：先将数据集$D$分成$k$个大小相似的互斥子集，即$D=D_1\cup D_2\cup \cdots \cup D_k, D_i \cap D_j = \emptyset(i \ne j)$,每次使用$k-1$个子集的并集作为训练集，余下的那个子集作为测试集，从而进行$k$此训练去平均值（通常又称$k$折交叉验证，$k$一般取10，其他常用取值5，20）

  ![1559280496714](C:\Users\CMD\AppData\Roaming\Typora\typora-user-images\1559280496714.png)关于数据集$D$的划分，为减少因样本划分不同而引入的差别，$k$折交叉验证通常要随机使用不同的划分重复$p$次，得到$p$次$k$折交叉验证

  **留一法(Leave-One-Out,LOO)**：令$k=m$，留一法不受随机样本划分方式的影响，评估的结果一般认为比较精准，但是当$m$特别的的时候，计算开销可能是难以忍受的

### 2.2.3 自助法

- **自助法(bootstrapping)**：能够使训练集和$D$更加相近，直接采用自助采样法(bootstrap sampling)，给定$m$个样本的数据集$D$,我们对它进行采样产生$D'$：每次随机从$D$中挑选一个样本拷贝放入$D'$中，该样本再放入$D$中，重复$m$次，得到包含$m$个样本的数据集$D'$，$D$中不采到的样本概率$（1-\frac{1}{m}）^m$,求极限得到
  $$
  \lim_{m \to \infty}(1-\frac{1}{m})^m\rightarrow\frac{1}{e}\approx0.368
  $$
  $D$中有36.8%的数据未出现在$D'$中，可以将$D'$作为训练集，D\D'作为测试集，亦称“外包估计(out-of-bag-estimate)”，适用于数据集较小，难以有效划分训练/测试集时用

### 2.2.4 调参与最终模型

- 给定的$m$个样本的数据集$D$，在模型评估与选择的过程中由于要留出一部分数据进行评估测试，所以只使用了一部分数据训练，在模型选择完成后，学习算法和参数配置已选定，此时应该用$D$重新进行训练
- **验证集(validation set)**：模型评估与选择中用于评估测试的数据集，把训练数据另外划分为训练集和验证集，基于验证集上的性能来进行对模型的选择和调参

## 2.3 性能度量

- **性能度量**：衡量模型泛化能力的评价标准

  在预测任务中，样例集$D=\{(x_1,y_1),(x_2,y_2),\cdots(x_m,y_m)\}$,其中$y_i$是示例$x_i$的真实标记，要评估学习器$f$的性能，就要将$f(x)$与真实标记$y$比较

  回归任务中最常用的性能度量是**均方误差**
  $$
  E(f;D)=\frac{1}{m}\sum_{i=1}^m(f(x_i)-y_i)^2
  $$
  更一般的，对于数据分布$D$和概率密度函数$P(\cdot)$，均方误差为
  $$
  E(f;D)=\int_{x \sim D}(f(x)-y)^2p(x)dx
  $$

### 2.3.1 错误率与精度

- 错误率
  $$
  E(f;D)=\frac{1}{m}\sum_{i=1}^{m}\prod(f(x_i)\ne y_i)
  $$

- 精度
  $$
  acc(f;D)=\frac{1}{m}\sum_{i=1}^{m}\prod(f(x_i = y_i))\\
  =1-E(f;D)
  $$

- 对于数据分布$D$和概率密度函数$p(\cdot)$
  $$
  E(f;D)=\int_{x \sim D}\prod(f(x)\ne y)p(x)dx
  $$

  $$
  acc(f;d) = \int_{x \sim D} \prod(f(x) = y)p(x)dx\\
  =1-E(f;D)
  $$

  

### 2.3.2 查准率 查全率与F1

- 对于二分类问题，可以划分为**真正例(true positive)  假正例(false positive)  真反例(true negative)  假反例(false negative)**,分别用**TP FP TN FN**表示，且$TP+FP+TN+FN=样例总数$ 

  ![1559290458756](C:\Users\CMD\AppData\Roaming\Typora\typora-user-images\1559290458756.png)

  查准率和查全率分别定义为
  $$
  P=\frac{TP}{TP+FP}
  $$

  $$
  R=\frac{TP}{TP+FN}
  $$

  查准率和查全率是一对矛盾的度量，可以用P-R曲线描述
  
  ![1559543399356](C:\Users\CMD\AppData\Roaming\Typora\typora-user-images\1559543399356.png)
  
  当一个学习器的P-R曲线被另一个包住，则后者的性能优于前者，平衡点(Break-Event Point,BEP)作为查准率，查全率的性能度量，但是BEP过于简化，更常用的是$F 1$度量
  $$
  F1= \frac{2\times P\times R}{P+R}=\frac{2 \times TP}{样例总数+TP-TN}
  $$
  $F1$度量的一般形式$—F_\beta$
  $$
  F_\beta=\frac{(1+\beta^2)\times P\times R}{(\beta^2\times P)+R}
  $$
  其中$\beta >0$度量了查全率对查准率的相对重要性，$\beta=1$时则退化为标准的$F  1$,$\beta>1$时查全率有更大影响，$\beta<1$时查准率有更大影响

- **宏查准率与微查准率** 分别在各混淆矩阵上计算查准率与查全率,记为$(P_1,R_1),(P_2,R_2),\cdots,(P_n,R_n)$

  **宏查准率，宏查全率，宏$F1$ **
  $$
  macro-P=\frac{1}{n}\sum_{i=1}^{n}P_i\\
  macro-R=\frac{1}{n}\sum_{i=1}^{n}R_i\\
  macro-F1=\frac{2\times macro-P \times macro-R}{macro-P+macro-R}
  $$
  先对各混淆矩阵对于元素进行平均，得到$\overline{TP},\overline{FP},\overline{TN},\overline{FN}$,在基于这些平均值求出**微查准率，微查全率，微$F 1$ **
  $$
  micro-P=\frac{\overline{TP}}{\overline{TP}+\overline{FP}}\\
  micro-R=\frac{\overline{TP}}{\overline{TP}+\overline{FN}}\\
  micro-F1=\frac{2\times micro-P \times micro-R}{micro-P +micro-R}
  $$

### 2.3.3 ROC与AUC

- **ROC 受试者工作特征(Receiver Operating Characteristic)** 以真正例率（True Positive Rate，TPR）为纵轴，以假正例率（False Positive Rate，FPR）为横轴
  $$
  TPR=\frac{TP}{TP+FN}\\
  FPR=\frac{FP}{TN+FP}
  $$
  ![1559546159553](C:\Users\CMD\AppData\Roaming\Typora\typora-user-images\1559546159553.png)对角线对应于“随机猜测”模型，点$(0,1)$对应于将正例排在所有反例前面则对应于”完美模型“

  **性能度量** 若一个学习器的ROC被另一个学习器的曲线包住，则后者的性能优于前者，若交叉，则用AUC（Area Under ROC Curve）进行比较，AUC可以通过ROC曲线下的各部分面积求得，假设ROC曲线由坐标$\{(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)\}$,则AUC可估算为
  $$
  AUC=\frac{1}{2}\sum_{i=1}^{m-1}(x_{i+1}-x_i)\cdot(y_i+y_{i+1})
  $$
  给定$m^+$个正例和$m^-$个反例，令$D^+$和$D^-$分别表示正，反例集合，则排序"损失"（loss)定义为
  $$
  l_{rank}=\frac{1}{m^+m^-}\sum_{x^+ \in D^+}\sum_{x^- \in D^-}\parallel((f(x^+)<f(x^-))+\frac{1}{2}\parallel(f(x^+)=f(x^-)))
  $$
  即考虑每一对正、反例，若正例的预测值小于反例，则记一个”罚分“，若相等，则记0.5个”罚分“，$l_{rank}$对应的是ROC曲线之上的面积
  $$
  AUC=1-l_{rank}
  $$

### 2.3.4 代价敏感错误率与代价曲线

- 为权衡不同类型错误造成的不同损失，可为错误赋予”非均等代价“，以二分类任务为例，设定一个”代价矩阵“

  ![1559549409963](C:\Users\CMD\AppData\Roaming\Typora\typora-user-images\1559549409963.png)

  $cost_{ij}$表示将第$i$类样本预测为第$j$类样本的代价，$cost_{ii}=0$,若将第$0$类判别为第$1$类所造成的损失更大，则$cost_{01}>cost_{10}$ 

- **总体代价** 

  - ”代价敏感“错误率
    $$
    E(f;D;cost)=\frac{1}{m}(\sum_{x_i \in D^+} \parallel(f(x_i) \ne y_i)\times cost_{01}+\sum_{x_i \in D^-}\parallel(f(x_i)\ne y_i)\times cost_{10})
    $$
    

