# 南方科技大学-CS303_311-人工智能-大作业

#### 介绍

这是南方科技大学CS303/311人工智能课程的大作业——玩转黑白棋。This is the project of CS303/311 Artificial Intelligence course in Southern University of Science and Technology, which is to design AI to play Reverse Reversi. 

#### 软件架构

Python

#### 安装教程

1. xxxx
2. xxxx

#### 使用说明

1. xxxx
2. xxxx
#### 研发阶段

```mermaid
flowchart TD
subgraph A ["模型准备"]
 direction TB
 a1(文献综述)
 a2(定理推导)
 a3("同学讨论、请教老师和学助")
 
 a1---a2---a3
end
subgraph B ["模型假设"]
direction TB
 	b1(评价模型)
	b2(性能模型)
	b3(搜索模型)
	b1---b2---b3
end
subgraph C ["模型求解"]
direction LR
subgraph c1["软件基础架构开发"]
	direction TB
	c11(本地的高速模拟器)
	c12(本地的游戏GUI界面)
	c13(Python和Numba学习)
	c11---c12---c13
end
subgraph c2["AI模型实现"]
	direction TB
	c21(评价模型)
	c22(性能模型)
	c23(搜索模型)
	c21---c22---c23
end
subgraph c3["本地的软件测试"]
	direction TB
	c31("烟雾测试（设计的基础测试样例下可运行）")
	c32("回归测试（新算法不比旧算法弱\基础软件架构速度没有变慢）")
	c31---c32
end
c1-->c2-->c3
end
subgraph D ["模型分析"]
    direction TB
	d1(发现问题)
	d2(构想假说)
	d3(演绎假说)
	d4(实验验证)
	d1-->d2-->d3-->d4-->d2
	d5(游戏性质估计实验)
	%%subgraph d6 ["评估函数有效性实验"]d61(专家法特征)d62( ) %%end
	d6(评估函数有效性实验)
	d7(搜索策略有效性实验)
	d5---d6---d7
end
A-->B-->C-->D-->B
```



1. 基础软件架构
- 超时
- AI与模拟器与GUI分离
- Python语法探究与OJ部署
2. 评估函数调优
- 本地对抗实验
- 理论推导
- 演化计算
- 论文学习
3. 搜索算法调优
- 同函数不同深度的实验
- MCTS
- 论文学习
4. 搜索算法时间利用
- IDS
- MCTS 实验次数
- numba优化
- 论文学习
5. 参考文献生成
- citation machine
- zetero
- Markdown 学术

#### 参与贡献

1. Fork 本仓库

2. 新建 Feat_xxx 分支

3. 提交代码

4. 新建 Pull Request

   

#### 特技

1. 
