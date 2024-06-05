[toc]

# 短视频和广告推荐经验总结
总结一下目前工作中涉及到的短视频和广告推荐领域相关优化经验，个别地方表述未必准确，欢迎大家批评指正~

## 召回

## 粗排

## 精排

### 回归问题建模：时长建模，GMV建模，LTV建模

**短视频、广告及推荐场景经常涉及到对连续值的预估建模，如时长建模、GMV建模、流量ecpm预估、LTV预估、付费次数、直播打赏金额预估等，回归问题相比分类问题更加难以优化，原因如下：**

1. 回归问题的损失函数更难选取。
回归问题的损失函数一般都隐含着对样本概率分布的假设，如果真实的后验数据不符合该假设，拟合效果就会很差；而分类问题的损失函数如交叉熵，对标签、误差没有什么前置假设，仅仅等价于最大似然

2. 分类问题对误差的容忍更大。
例如，逻辑回归输出0.6和1.0是一样的，最后都会被转化为同一个类别。而回归则不是，0.1的误差就是0.1。因此，回归问题对结果的要求更严格，同时，这也导致回归问题对离群点更为敏感。
从优化的角度讲，由于回归问题对离群点敏感，这就导致在梯度下降（或反向传播）时容易发生梯度消失或梯度爆炸的问题。同时，由于回归问题对误差要求比较高，过多的正则会导致结果变差，因此一些在分类问题里面常用的正则化方法无法简单的套用在回归问题中。这使得回归问题调参变得困难。
从样本的角度看，由于回归问题对误差的要求比较高，因此对样本量的需求也更大。例如，对于两个样本，其真值分别为0.6和1.0，因此它们是属于同一类的，对于分类问题，学习器只要学习到它们属于同一类即可，因此不要求那么多的数据，而对于回归问题，有时0.1的误差都是很大的，因此对样本量的需求也会更大。

3. 样本不均衡。
分类问题中，样本不均衡是指一个或多个类的样本数量远远少于其他类，导致模型无法很好的学习这些类的信息，进而使模型更倾向于预测样本多的类。在CV中，常采用的方法是augmentation，旋转平移等方法增加样本量。在回归问题中，不仅仅是不均衡，而是经常出现“断片”现象。如一个问题的值域是[0, 100]，但是在[10, 100]里样本量很少，几乎没有样本，不能硬插值填充数据，这种情况会导致严重的样本不均衡问题，进而在该区间上拟合效果非常差。
另外，回归问题的值域可能是无穷的，而样本所对应的空间仅仅是很小的一部分。当新样本的label不在训练样本对应的值域中时，模型的预测值也会严重偏离实际。

参考：https://cloud.tencent.com/developer/news/60043

**如何对回归问题进行优化，主要有以下优化思路：**

1. 损失函数优化

优化方向：损失函数上做优化，不同的损失对数据有不同的先验分布假设，针对业务场景做出适配调整。


| 损失函数 | insight | 推导 | 
| :----:| :----: | :----: |
| MSE | error（即label-pred）服从[正态分布](https://zh.wikipedia.org/wiki/%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83) | 假设 $y_{i} = h_{\theta}(x_{i};\theta) + \epsilon_{i}$, 其中 $\epsilon_{i}$ 为error，假设erro服从高斯分布: $\epsilon_{i} \sim \mathcal{N}(0, \sigma^2)时$, 其概率密度函数为: $p(\epsilon_{i}) = \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{\epsilon_{i}^2}{2\sigma^2})$ 由误差定义，可得: $p(y_{i} \| x_{i}; \theta) = \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y_{i} - h_{\theta}(x_{i};\theta))^2}{2\sigma^2})$, 再用最大似然原理最大化上式，可得: $$log(L(\theta)) = log \prod_{i=1}^{N} p(y_{i} \| x_{i}; \theta) = Nlog\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2}\sum_{i=1}^{N} (y_{i} - h_{\theta}(x_{i};\theta))^2$$ $$\Leftrightarrow min \frac{1}{2} \sum_{i=1}^{N} (y_{i} - h_{\theta}(x_{i};\theta))^2$$ |
| MAE | error（即label-pred）服从[拉普拉斯分布](https://zh.wikipedia.org/wiki/%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%E5%88%86%E5%B8%83) | 假设 $y_{i} = h_{\theta}(x_{i};\theta) + \epsilon_{i}$, 其中 $\epsilon_{i}$ 为error，假设erro服从拉普拉斯分布: $\epsilon_{i} \sim \mathcal{L}(0, 1)时$, 可得: $p(y_{i} \| x_{i}; \theta) = \frac{1}{2}exp(-\|y_{i} - h_{\theta}(x_{i};\theta)\|)$, 再用最大似然原理最大化上式，可得: $$log(L(\theta)) = log \prod_{i=1}^{N} p(y_{i} \| x_{i}; \theta) = Nlog\frac{1}{2} - \sum_{i=1}^{N} \|y_{i} - h_{\theta}(x_{i};\theta)\|$$ $$\Leftrightarrow min \sum_{i=1}^{N} \|y_{i} - h_{\theta}(x_{i};\theta)\|$$ |
| Huber Loss | Huber loss是MAE和MSE损失函数的结合，具备两者的优点 |
| Huberpp Loss | MSE/MAE/Huber Loss关注的是绝对误差，而在业务中，关注的更多的是相对误差 |
| AM Loss | 以上Loss均是对称性Loss，而在业务中，我们对于高低估问题关注程度不同，可以据此调整 |


2. 回归转分类
优化方向：采用各类变换方式将回归问题转为分类问题进行优化。





3. calibration


#### MSE Loss

$$Loss=- \frac{1}{N} \sum_{i=1}^N (label_{i} - pred_{i})^2$$

用MSE损失函数对观看时长 or GMV or LTV 拟合，由于观看时长 or GMV or LTV 数值范围较大，直接对原始值拟合会导致 $logit$ 出现较大的range，导致模型陷入局部最优，引起梯度爆炸。

对观看时长 or GMV or LTV 做 $log$ 或 $e^{0.3}$ 变换后进行拟合，由于变换函数不是线性的，该方法拟合是有偏估计，对于数值较大的样本拟合时会出现低估现象。

**MSE隐含的假设是error服从正态分布，然而很多情境下这个假设并不成立。当error不服从正态分布时，MSE的效果就有可能受损。**

#### MAE Loss

$$Loss=- \frac{1}{N} \sum_{i=1}^N |label_{i} - pred_{i}|$$

**MAE隐含的假设是error服从拉普拉斯分布，然而很多情境下这个假设并不成立。当error不服从拉普拉斯分布时，MAE的效果就有可能受损。**

#### Huber Loss
Huber Loss是MAE和MSE损失函数的结合, $\delta$ 的大小决定了损失函数对MAE和MSE的侧重程度：

$$Loss=- \frac{1}{N} \sum_{i=1}^N Loss_{i}$$

$$Loss_{i} = \begin{cases}
\frac{1}{2} (pred_{i}-label_{i})^2, & if \ |pred_{i}-label_{i}|<= \delta \\
\delta |pred_{i}-label_{i}| - \frac{1}{2}\delta^2, & if \ |pred_{i}-label_{i}|>\delta \\
\end{cases}$$

求梯度为

$$|\frac{\partial Loss} {\partial pred}| = \begin{cases}
|pred-label|, & if \ |pred-label|<= \delta \\
\delta, & if \ |pred-label|>\delta \\
\end{cases}$$

即 $|\frac{\partial Loss} {\partial pred}|$ 先随着绝对预估偏差 $|pred-label|$ 线性增长, $|pred-label|$ 超过 $\delta$ 后就封顶，因此Huber Loss既能赋予绝对预估偏差大的样本更大的梯度更新参数，也能限制最大不超过 $\delta$ 预防极端异常值。

#### Huberpp Loss

广告出价系统里，GMV模型预估值直接参与ROI出价，相对预估偏差是GMV模型更加关注的指标。如GMV模型将5元的样本预估为10元，将100元的样本预估为105元，从Huber Loss来看，两者得到的Loss绝对值和梯度步长项均一样，但对于广告竞价来看，前者的预估偏差和超成本风险要远高于后者，因此可以对Huber Loss做调整，让GMV模型更加关注相对预估偏差更大的样本。我们希望 $|\frac{\partial Loss} {\partial pred}|$ 随着相对预估偏差 $\frac{|pred - label|}{label}$ 线性增长，且 $\frac{|pred - label|}{label}$ 超过 $\delta$ 后就封顶，公式如下：

$$|\frac{\partial Loss} {\partial pred}| = \begin{cases}
C \cdot \frac{|pred - label|}{label}, & if  \ \frac{|pred - label|}{label}<= \delta \\
C \cdot \delta, & if \  \frac{|pred - label|}{label}>\delta \\
\end{cases}$$

其中 $\delta$ 和 $C$ 是超参数, $\delta$ 决定相对值达到多少时步长封顶, $C$ 决定分段函数前半段 $|\frac{\partial Loss} {\partial pred}|$ 随着 $\frac{|pred - label|}{label}$ 线性增长速度, $C \cdot \delta$ 是封顶步长，反推得到：

$$Loss_{i} = \begin{cases}
\frac{C}{2 \cdot label_{i}} (pred_{i}-label_{i})^2, & if \ \frac{|pred_{i} - label_{i}|}{label_{i}}<= \delta \\
C \cdot \delta |pred_{i}-label_{i}| - \frac{C \cdot \delta^2 \cdot label}{2}, & if \ \frac{|pred_{i} - label_{i}|}{label_{i}}>\delta \\
\end{cases}$$

后半段的常数项 $- \frac{C \cdot \delta^2 \cdot label}{2}$ 是保证Loss在分段函数断点处连续。

#### Asymmetric Modeling loss，AM loss
MAE、MSE、Huber Loss和Huberpp Loss在高低估的时候loss计算相同，即不满足非对称性，而在GMV预估中，更希望模型优先优化低估的问题，提高收入上限。设计新的loss函数，采用log函数对相对误差做平滑，且当 $pred = label$ 时，梯度为0，如下所示：

$$ \frac{\partial Loss} {\partial pred} = ln \frac{pred} {label} $$

在不同的 $label$ 区间下，可以对梯度增加额外的rescale，以满足各区间个性化需求；另外，样本里会存在一些极端case，如低估99%、高估几千倍，可以根据相对预估偏差对梯度上限做截断，预估值限制在 $[lowBound * label, upBound * label]$, 简称 $[LB * label, UB * label]$, 公式如下：

$$\frac{\partial Loss} {\partial pred} = \begin{cases}
C \cdot S_{h} \cdot ln(UB), & if \ pred > UB * label \\
C \cdot S_{h} \cdot ln(\frac{pred}{label}), & if \ UB * label >= pred > label \\
C \cdot S_{l} \cdot ln(\frac{pred}{label}), & if \ LB * label <= pred <= label \\
C \cdot S_{l} \cdot ln(LB), & if \ pred < LB * label \\
\end{cases}$$

由导数反向推导损失函数，并考虑分段函数的连续情况，如下：

$$Loss = \begin{cases}
C \cdot S_{h} \cdot [ln(UB) \cdot |label - pred| - ln(UB) \cdot (UB-1) \cdot label + UB \cdot label \cdot (ln(UB)-1)] - (S_{l} - S_{h}) \cdot C \cdot label, & if \ pred > UB * label \\
C \cdot S_{h} \cdot pred \cdot [ln(pred) - ln(label) -1] - (S_{l} - S_{h}) \cdot C \cdot label, & if \ UB * label >= pred > label \\
C \cdot S_{l} \cdot pred \cdot [ln(pred) - ln(label) -1], & if \ LB * label <= pred <= label \\
C \cdot S_{l} \cdot [-ln(LB) \cdot |label - pred| + ln(LB) \cdot (1-LB) \cdot label + LB \cdot label \cdot (ln(LB) - 1)], & if \ pred < LB * label \\
\end{cases}$$


#### log normal(ZILN)


#### WCE加权分类

- 优化目标

建模用户观看短视频的时长

- 方案借鉴

参考YouTube的[Deep Neural Networks for YouTube](Recommendationshttps://dl.acm.org/doi/pdf/10.1145/2959100.2959190)，是否点击作为正负样本，将正样本的观看时长作为其样本权重，负样本权重为1，用加权逻辑回归进行训练，线上infer用odds作为预估时长，具体推导过程参考：https://zhuanlan.zhihu.com/p/435912211

- WCE加权分类

普通二值交叉熵损失函数定义如下：
$$Loss=- \frac{1}{N} \sum_{i=1}^N [y_{i} \cdot log(p_{i}) + (1-y_{i}) \cdot log(1-p_{i})] $$
WCE(加权交叉熵损失函数)定义如下：
$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot y_{i} \cdot log(p_{i}) + (1-y_{i}) \cdot log(1-p_{i})]$$

这和普通二值交叉熵仅仅有一点变化，就是在**正样本**的判别上加了一个 $w_{i}$ 系数，而该系数的设定则有很多方法。

- WCE具体方案

Youtube的页面是上下滑的列表，因此是否点击就可以作为正负样本的区分，但是短视频是沉浸式业务，每个视频都是点击的状态，都有播放时长，正负样本的定义较难界定，因此业界的做法是：用样本的观看时长作为正样本的权重，同时复制一份样本作为负样本，采用加权Logloss训练，因此 $odds$ 的表征会做如下调整：

$$ odds = \frac{p}{1-p} = \frac{\sum_{i=1}^k T_{i}}{N - k + k} = \frac{\sum_{i=1}^k T_{i}}{N} = E(T) $$

具体实现方案有两种：
1. 直接修改loss函数
$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(p_{i}) + log(1-p_{i})]$$

此时，正样本的梯度变为原来的 $w_{i}$ 倍，即 
$$w_{i}*(1-p_{i})$$

负样本的梯度保持不变，即 
$$1*(0-p_{i})$$

样本梯度推导见：https://lukebest.github.io/posts/d28f

2. 样本复制一份作为负样本，权重为1，然后一起训练

两种方案除了工程性能的差异，最终的训练效果是否有差异？？？

- 短视频WCE具体实现方案

WCE建模时长的方案在国内某手app上得到充分验证，对于权重 $w_{i}$ 有以下3种方案：

|  | 方案1 | 方案2 | 方案3 |
| :----:| :----: | :----: | :----: |
| 观看时长 $wt$ | $wt \in (0, +\infty]$ | $wt \in (0, +\infty]$ | $wt \in (0, +\infty]$ |
| 正样本权重 $w$ | $w_{1} = \log_2 (wt+1)$ | $w_{2} = \log_2(wt+1)+1$ | $w_{3} = wt$ |
| 负样本权重 | 1 | 1 | 1 |
| 模型预测值 <br> 模型输出经过sigmoid变换 | $p_{1} = \frac{1}{1+e^{-\theta_{1} x}}$| $p_{2} = \frac{1}{1+e^{-\theta_{2} x}}$ | $p_{3} = \frac{1}{1+e^{-\theta_{3} x}}$ | 
| $odds$ | $odds = \frac{p_{1}}{1-p_{1}} = e^{\theta_{1} x} = w_{1}$ <br> $p_{1} = \frac{w_{1}}{w_{1} +1}$ | $odds = \frac{p_{2}}{1-p_{2}} = e^{\theta_{2} x} = w_{2}$ <br> $p_{2} = \frac{w_{2}}{w_{2} +1}$ | $odds = \frac{p_{3}}{1-p_{3}} = e^{\theta_{3} x} = w_{3}$ <br> $p_{3} = \frac{w_{3}}{w_{3} +1}$ |
| 模型预测值还原为时长 | $wt = 2^{\frac{p_{1}}{1-p_{1}}} -1 $ | $wt = 2^{(\frac{p_{2}}{1-p_{2}} -1)} -1$ | $wt = \frac{p_{3}}{1-p_{3}}$ |
| 变量边界条件 | $wt \in (0, +\infty]$ <br> $w_{1} \in (0, +\infty]$ <br> $p_{1} \in (0, 1]$ | $wt \in (0, +\infty]$ <br> $w_{2} \in (1, +\infty]$ <br> $p_{2} \in (\frac{1}{2}, 1]$ | $wt \in (0, +\infty]$ <br> $w_{3} \in (0, +\infty]$ <br> $p_{3} \in (0, 1]$ |
| 时长边界条件 | $当wt=1时，$ <br> $w_{1} = 1$ <br> $p_{1} = \frac{1}{2}$ | $当wt=1时，$ <br> $w_{2} = 2$ <br> $p_{2} = \frac{2}{3}$ | $当wt=1时，$ <br> $w_{3} = 1$ <br> $p_{3} = \frac{1}{2}$ |
| 模型预测值图示 | ![1.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/1.png) | ![2.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/2.png) |

对于权重 $w$，业界有两种处理方法：

一种是对其做log变换，如方案1和2所示，快手采用该处理方法：观看时长 $wt \in (0, +\infty]$，该值可以无限大，但 $w$ 不能无限大，因此会对 $w = \log_2 (wt+1)$ 设置阈值上界，经验值设置为8，原因？

一种是直接用原值，如方案3所示：此方法与YouTube的论文和字节的处理方式一致

- WCE隐含的假设是y服从几何分布 $p(y) = p^y \cdot (1-p)$ 最大似然可得wce loss，即wce的预估值等于数学期望，是无偏预估。以方案3为例，具体推导如下：

$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(p_{i}) + log(1-p_{i})]， p_{i} = \frac{1}{1 + e^{-\theta x}}$$

求解该损失，得到无偏的预估时长 $\widehat{wt_{i}} = \frac{p_{i}}{1-p_{i}} = e^{\theta x_{i}}$ ，用预估值来表示损失函数可写为：

$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(\frac{\widehat{wt_{i}}}{1+\widehat{wt_{i}}}) + log(1-\frac{\widehat{wt_{i}}}{1+\widehat{wt_{i}}})]$$

$$ =- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(\widehat{wt_{i}}) - (1+w_{i}) \cdot log(1+\widehat{wt_{i}})]$$

$$ =- \frac{1}{N} \sum_{i=1}^N log[\widehat{wt_{i}}^{w_{i}} \cdot (1+\widehat{wt_{i}})^{-(1+w_{i})}] $$

$$ =- \frac{1}{N} \sum_{i=1}^N log[(\frac{\widehat{wt_{i}}}{\widehat{wt_{i}}+1})^{w_{i}} \cdot \frac{1}{\widehat{wt_{i}}+1}] $$

当 $w_{i} \in (0,1,2,..., +\infty]$，令 $p_{i} = \frac{1}{\widehat{wt_{i}}+1}$，有 $P(w_{i}=k|x_{i}) = (1-p_{i})^k \cdot p_{i}$，其满足几何分布，数学期望为 $\frac{1 - p_{i}}{p_{i}} = \widehat{wt_{i}}$ ，即WCE的预估值等于数学期望，是无偏预估。

几何分布的期望和方差如下：
![6.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/6.png)

几何分布：https://en.wikipedia.org/wiki/Geometric_distribution

WCE的缺点：
- WCE隐含的假设是y服从几何分布。虽然WCE是无偏预估，预估值等于数学期望，但其隐含的假设是y服从几何分布，如果label的分布和几何分布差异较大，则WCE效果会变差；
- WCE在低估和高估的时候梯度大小不同。对loss梯度做简单推导后可以发现WCE在低估和高估的时候梯度大小不同，具体推导如下：

$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(\widehat{wt_{i}}) - (1+w_{i}) \cdot log(1+\widehat{wt_{i}})]$$

$$ \frac {\partial Loss} {\partial \widehat{wt_{i}}}=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot \frac{1}{\widehat{wt_{i}}} - (1+w_{i}) \cdot \frac{1}{1+\widehat{wt_{i}}}]$$

$$ = \frac{1}{N} \sum_{i=1}^N \frac{\widehat{wt_{i}} - w_{i}}{\widehat{wt_{i}} \cdot (1+\widehat{wt_{i}})}$$

以 $w_{i}=20$ 为例，画出梯度图像如下所示：
![4.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/4.png)

可以看出，在低估 $\widehat{wt_{i}}<20$ 和高估 $\widehat{wt_{i}}>20$ 时，梯度具有严重的不对称性。低估时，梯度较大；高估时，梯度较小。

#### softmax多分类
将观看时长做分桶离散化，进而将回归问题转为多分类问题，业界方案如下：

- 对观看时长 $wt$ 做等频划分，划为 $K$ 个桶：

$$ [wt_{1},wt_{2},wt_{3}...,wt_{K}] $$
$$ wt_{k}为桶的边界值，k=1,2,3..,K$$

- 根据样本的观看时长得到label转换后的类别：

当 $wt_{k-1} < wt_{i} < wt_{k}$时， 该样本转换后的类别为 $b_{i}$

- 模型优化由回归变为多分类问题，loss函数如下：

$$Loss=- \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K I(b_{i}) \cdot log(p_{i,k})$$

$$I(b_{i})只在对应类别为1，其余为0$$

- 线上serving时，得到样本的预测时长

$$ \widehat{wt_{i}} = \sum_{k=1}^K m_{k} \cdot p_{i,k} $$

$$ m_{k}是第k个桶的均值或中值，p_{i,k}表示样本i预测时长是第k类的概率 $$

**softmax多分类采用的是0-1的hard label，目标只能离散到一个桶里，类别之间绝对隔离，这种硬标签忽略了负标签直接的差别，丢失了回归问题label连续的假设。例如，某样本被分到第 $k$ 个桶， 损失函数只区分了预估为 $k$ 或者不为 $k$，当不为 $k$ 时，预测为 $k+1$ 或者 $k+n$ 的损失都为0，没有区别。但是作为回归问题，分桶值大小有顺序的含义，预测为 $k+n$ 的损失应该大于 $k+1$ 的损失。**

#### distill softmax多分类

distill softmax多分类借鉴了知识蒸馏中soft label的思路，依然使用原始label分桶后做模型softmax输出做多分类的思路。假如我们有一个很强的teacher时长模型，输出应该是原始label附近的一个概率分布，我们的student模型去学习这个概率分布，这个概率分布大概率不是一个等方差的高斯分布，而是一个label_aware的分布，更贴近label的误差分布。这样既能让student模型捕捉到label间的关系，也能降低student模型的学习难度，模型最终使用输出的概率分布做加权平均得到最终的staytime预估值。

实际上我们并没有teacher模型，但反过来思考下我们需要teacher模型原因是对原始label分桶的一个软化过程，只是这个软化使用的是能区分样本难度的teacher模型来做。如果只是考虑软化，基于人工经验我们是可以做的相当好的，可以跳过训练teacher模型这一步，对原始label分类软化的这一步我们称为人工蒸馏，人工蒸馏中软化还可以做到分布均值的无偏以及分布的label_aware，甚至比强teacher模型做得更多更好。

最终使用人的先验知识作为虚拟teacher对时长label进分桶行了软化，帮助多分类任务感知label间序关系以及分布，降低了时长模型的学习难度。

假设时长分桶服从某个先验分布 $p(wt_{k})$，可以用KL散度来学习预测分布 $p_{i,k}$ 和 $p^\prime_{i}(wt_{k})$ 的相似性，即：

$$Loss=\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p^\prime_{i}(wt_{k}) \cdot log\frac{p^\prime_{i}(wt_{k})}{p_{i,k}}$$

$$=\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p^\prime_{i}(wt_{k}) \cdot [log(p^\prime_{i}(wt_{k})) - log(p_{i,k})]$$

$$=\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p^\prime_{i}(wt_{k}) \cdot log(p^\prime_{i}(wt_{k})) - p^\prime_{i}(wt_{k}) \cdot log(p_{i,k})$$

$$=-\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p^\prime_{i}(wt_{k}) \cdot log(p_{i,k}) + const$$


$$当 p(wt_{k}) \sim \mathcal{N}(wt_{i}, \sigma^2)时，p_{i}(wt_{k}) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}(\frac{wt_{k} - wt_{i}}{\sigma})^{2}}，p^\prime_{i}(wt_{k}) = \frac{p_{i}(wt_{k})}{\sum_{k=1}^Kp_{i}(wt_{k})}$$ 

$$当 p(wt_{k}) \sim Laplace(wt_{i}, \sigma)时，p_{i}(wt_{k}) = e^{-\frac{|wt_{k} - wt_{i}|}{\sigma}}，p^\prime_{i}(wt_{k}) = \frac{p_{i}(wt_{k})}{\sum_{k=1}^Kp_{i}(wt_{k})}$$ 

$$ wt_{i}是样本i的观看时长，为真实label值，wt_{k}是时长分桶第k个桶的桶内均值或者桶边界，此处以桶边界为例 $$

$$ \sigma 为超参数，可以设置为定值，也可以label_aware，如 \sigma = 1.5 \cdot \sqrt{wt}，label越大，\sigma 越大，可根据后验数据分析确定 \sigma 的分布形态 $$

$$ p_{i}(wt_{k})是计算样本 {i} 在各个分桶上的分布， p^\prime_{i}(wt_{k})为归一化的概率值，这样保证了样本 {i} 在各个分桶上的概率加和=1，也符合多分类softmax的定义。 $$

![5.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/5.png)

软化函数的设计则只要label变换后的分布符合场景后验值的分布即可。
无论哪种软化策略，都依赖分桶，顾名思义是对staytime的一堆边界划分点，可以均匀划分，也可以非均匀划分，具体每个场景的分桶策略可根据场景特性决定，核心原则是让桶内的样本数量分布更均匀。

- 线上serving时，得到样本的预测时长

$$ \widehat{wt_{i}} = \sum_{k=1}^K m_{k} \cdot p_{i,k} $$

$$ m_{k}是第k个桶的均值或中值，p_{i,k}表示样本i预测时长是第k类的概率 $$

#### 分桶LogLoss建模/ordinal regression建模

该方法来源CV的单目深度估计，将回归问题的预估转化为序数回归，具体见：

https://arxiv.org/pdf/1806.02446

https://github.com/hufu6371/DORN

论文的阅读笔记：https://blog.csdn.net/u012348774/article/details/110715870

简单的多分类存在没有考虑非目标类的之间的序关系的问题，Ordinal Regression则是一种考虑类间序关系的回归方法，推导过程参考：https://zhuanlan.zhihu.com/p/573572151

![3.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/3.png)

具体做法如下：

- 观看时长 $wt$ 划为 $K$ 个桶：

$$ [wt_{1},wt_{2},wt_{3}...,wt_{K}] $$
$$ wt_{k}为桶的边界值，k=1,2,3..,K$$

- 根据样本的观看时长得到转换后的label：

当 $wt_{k-1} < wt_{i} < wt_{k}$时，即 $b_{i}=k$，该样本转换后的label为：[1, 1, ..., 1, 0, ..., 0]

![8.jpg](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/8.jpg)

$wt_{i}$ 的样本在 $[wt_{k-1}, wt_{k}]$ 下类别为1，在 $[wt_{1}, wt_{2}]，[wt_{2}, wt_{3}]，...，[wt_{k-2}, wt_{k-1}]$ 下类别为1，在 $[wt_{k}, wt_{k+1}]，[wt_{k+1}, wt_{k+2}]，...，[wt_{K-1}, wt_{K}]$ 下类别为0


- 模型优化由回归问题转化为多个独立的二分类问题，loss函数如下：

$$Loss=- \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K I(k \le b_{i}) \cdot log(p_{i,k})$$

$$I(k \le b_{i})表示小于等于类别b_{i}时为1，其余为0$$

- 线上serving时，得到样本的预测时长

$$ \widehat{wt_{i}} = \sum_{k=1}^K I(k \le b_{i}) \cdot (m_{k} - m_{k-1}) \cdot p_{i,k} $$

$$ = 1 \cdot (m_{1} - m_{0}) \cdot p_{i,1} + 1 \cdot (m_{2} - m_{1}) \cdot p_{i,2} + ... + 1 \cdot (m_{b_{i}} - m_{b_{i}-1}) \cdot p_{i,b_{i}}  + 0 \cdot (m_{k+2} - m_{k+1}) \cdot p_{i,k+1} + ... + 0 \cdot (m_{K} - m_{K-1}) \cdot p_{i,K} $$

$$ = 1 \cdot (m_{1} - m_{0}) \cdot p_{i,1} + 1 \cdot (m_{2} - m_{1}) \cdot p_{i,2} + ... + 1 \cdot (m_{b_{i}} - m_{b_{i}-1}) \cdot p_{i,b_{i}} $$

$$ b_{i}=k表示样本i的时长lable是k，m_{k}是第k个桶的均值或中值，m_{0} = 0 $$

$$ p_{i,k}表示样本i预测时长是第k个桶的概率，I(k \le b_{i})表示小于等于类别b_{i}时为1，其余为0$$

softmax/distill softmax 和 ordinal regression都依赖分桶策略，分桶策略可根据场景特性决定，核心原则是让桶内的样本数量分布更均匀。下面给出一些常见的划分例子：

- uniform，设定bin_size，进行等距分桶
```python
ST_BOUNDS = tf.range(0.0, 3600.0, 1.0, dtype=M.get_dtype())
```

- segment，根据时长分布，手动分桶
```python
# segment
ST_BOUNDS = np.concatenate(
    [
        np.arange(0, 10, 0.2),
        np.arange(10, 180, 1.0),
        np.arange(180, 600, 10.0),
        np.arange(600, 3600, 50.0),
    ], axis=0
)
```

- exp_bin
时长场景建模推荐该种分桶方式，时长越短，数据更密集，分桶要更细；时长越长，数据更稀疏，分桶要更粗

```python
ST_BOUNDS = [np.exp(x/40.0) - 1 for x in range(bucket_size)]
```

电商场景GMV优化，分桶策略采用GMV分布的离散峰值作为桶边界进行分桶，软化函数采用log_norm的正态分布或者log_norm的拉普拉斯分布，类似zlin的log_norm，即先对label和bounds做log变换，然后做正态变换或者拉普拉斯变换。


- SID分桶策略，来源https://arxiv.org/pdf/1806.02446

$\alpha = 1, \beta = 500, K = 60$

```python
ST_BOUNDS = [np.exp(np.log(1) + np.log(500/1) * x / bucket_size) for x in range(bucket_size)]
```

#### D2Q
待补充
https://arxiv.org/pdf/2206.06003



**回归问题预估为什么会「高区间低估，低区间高估」？**

回归模型的预估总是会出现「高区间低估，低区间高估」的现象，而不是「低区间低估，高区间高估」，或是「无序混乱地低估/高估」呢？

- 定性解释
直观想象一下，假设我们要使用一个函数 $f(x)$ 来拟合如图所示的蓝色点，我们必然不可能让函数经过每一个点，一种较好的拟合方式如红线所示，对于局部点误差较大，但对整体点平均误差最小。在这种情况下，对于分布在红线左右的点，红线其实拟合的是他们的中间状态(可能是均值/中位数等，取决于使用的误差定义)，因此预估输出表现出的现象即为「高区间低估，低区间高估」。

![7.jpg](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/7.jpg)


通用的回归损失函数的收敛目标一般是样本点的均值或中值，这就导致了预估分布相比原始分布更加向“中心”靠拢，从而导致了「高区间低估，低区间高估」。

假设我们要寻找一个 $\beta$ , 使得当所有预估值都为 $\beta$ 时, $Loss$ 最小( $Loss$ 导数为0)

| Loss | 结论 | 推导 |
| :----:| :----: | :----: |
| MSE | MSE是均值回归 | $$Loss=\frac{1}{N} \sum_{i=1}^N (label_{i} - pred_{i})^2$$ , 代入 $\beta$ 可得 $$Loss=\frac{1}{N} \sum_{i=1}^N (label_{i} - \beta)^2$$ 于是有 $$\frac{\partial Loss} {\partial \beta} =\frac{2}{N} \sum_{i=1}^N (label_{i} - \beta)$$ 要求 $$\frac{\partial Loss} {\partial \beta}=0$$ , 则有 $$\beta=\frac{1}{N}\sum_{i=1}^N label_{i}$$|
| MAE | MAE是中值回归 | $$Loss=\frac{1}{N} \sum_{i=1}^N \|label_{i} - pred_{i}\|$$ , 代入 $\beta$ 可得 $$Loss=\frac{1}{N} \sum_{i=1}^N \|label_{i} - \beta\|$$ 于是有 $$\frac{\partial Loss} {\partial \beta} =-\frac{1}{N} \sum_{i=1}^N signal(label_{i} - \beta)$$ $$when \ label_{i} > \beta, signal(label_{i} - \beta) = 1;$$ $$when \ label_{i} < \beta, signal(label_{i} - \beta) = -1$$ 当 $label_{i} - \beta$ 正数和负数的数量相同时，也就是 $\beta$ 为 $label_{i}$ 的中位数时，该导数为0，Loss最小|
| WCE | WCE是均值回归 | $$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(\widehat{wt_{i}}) - (1+w_{i}) \cdot log(1+\widehat{wt_{i}})]$$ 代入 $\beta$ 可得 $$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(\beta) - (1+w_{i}) \cdot log(1+\beta)]$$ 于是有 $$\frac{\partial Loss} {\partial \beta} =-\frac{1}{N} \sum_{i=1}^N (\frac{w_{i}}{\beta} - \frac{w_{i}+1}{\beta+1})$$ 要求 $$\frac{\partial Loss} {\partial \beta} = 0$$ 则 $$\sum_{i=1}^N \frac{w_{i} - \beta}{\beta \cdot (\beta+1)} = 0$$ $$\beta=\frac{1}{N}\sum_{i=1}^N w_{i}$$ |


## 用户冷启动优化
待补充
PPNet
POSO





