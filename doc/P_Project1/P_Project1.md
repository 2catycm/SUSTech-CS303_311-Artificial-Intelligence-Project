# AI Project1: Development of a superb AI Program for Reversed Reversi 

叶璨铭，12011404@mail.sustech.edu.cn

[TOC]

## Introduction

In this project, we tries to develop a **high-level *Reversed Reversi* program** that are compatible to beat our classmates' programs. **Our purpose is not seeking to prevail over others, but to master the essence of AI system development by learning from and communicating with each other.**

 As Kronrod says, computer board game is the "fruit fly in Artificial Intelligence".[^1] This interesting computer game project will definitely provide me with **a better understaning** of the **knowledge I learnt in the AI course** via my **hand-by-hand practice and experiments** in the procedure of accomplishing this project. 

### Problem background

**Reversi**, also called Othello, is a deterministic, turn-taking, two-player, zero-sum game of perfect information.[^1]Reversi is not popular in China before the development of the Internet.[^6] Although it may not often appears as a board game, it is indeed popular in the research of computer game because of its relatively small search space. [^1]Computers have always excelled in Reversi because average human players cannot envision the drastic board change caused by move[^10], and because human players dislike to risk taking a seemingly bad but actually best move.[^7] 

**Reversed Reversi,** also called the anti mode of Othello, shares the same dynamics of the chessboard enviroment as Reversi, while the objective is the opposite. [^7]In the game, each player places a piece of his color on the board, flipping any opponent's pieces that are bracketed along a line.  The object of the **Reversi** is to have the **most discs** turned to display your color when the last playable empty square is filled, while **Reversed Reversi** expects the winner to have the **least discs.**[^8] A formal definition of the game rules of Reversed Reversi will be presented in Part 2.

### Literature review

Before we start to develop our own algorithms for Reversed Reversi, it is necessary for us to do a literature review on how previous researchers develop programs that play Reversi, Reversed Reversi and other board games. 

#### Normal Reversi program 

The first world class Reversi program is *IAGO* developed by Rosenbloom in 1982. This program effectively quantified Reversi maxims into efficient algorithms with adversial search techniques. [^10]Later in 1990, 李开复 developed another program called *BILL*, which far surpassed the *IAGO* by introducing dynamic evaluation function based on Bayesian learning. [^10][^2]Although *IAGO* and *BILL* are best computer programs that play Reversi at their times, the top human players are not beaten untill the occurence of the program *Logistello* developed by Buro in 1997.[^1]The main new idea of *Logistello* is that it automatically learns from previous plays[^6][^7]**After that, it is generally acknowledged that humans are no match for computers at Reversi.**[^1] 

In 1997-2006, Andersson developped a practical Reversi program called *WZebra*. It has big opening books, state-of-the-art search algorithms, pattern-based evaluation scheme that can stably run on Windows 2000 to even today's Windows 11 platforms. [^11]While it gains a better performace and stability by applying several techniques of C Programming Launguage, the basic ideas of *WZebra*, however, are no more than *BILL*'s or *Logistello*'s.  

#### Reversed Reversi program

While it is often the case that reversed board game is easy and boring, such as reversed  Chinese Chess, Go and Chess,   Reversed Reversi is worth playing and it is an art to play it well. [^7]According to MacGuire, much of the strategic thinking behind the classic game can also be applied to the reverse game, though sometimes in reverse.[^9]

Tothello, a program developed by Pittner, is believed to be the best  program in the world playing Reversed Reversi untill 2006. [^7]

#### General board game program

## Preliminary

In the last part, we have known the background of the problem and found some useful references. Next, we need to **formulate** the problem in formal language to **disambiguate the potential confusion.**[^8]With the formal logic system, we can then **derive some basic theorems and corollaries** that any Reversed Reversi game must logically follows. With these knowledge in our minds, it is clear how to design our models and experiments in the next sections.

### Problem formulation and notations

Informally, the Reversed Reversi problem is to build a program playing Reversed Reversi with some kinds of high *Intelligence*. 

Formally, this problem can be formulated as a **Task Enviroment,** which is specified by a tuple $(P, E, A, S)$, where P is the performance measure, E is the enviroment, A is the actuators, and S is the sensors. 

Besides problem, we also need to formulate the program. The program for this problem can be formulated as an **Agent**, which is specify by a function G, mapping the agents' percept histories to its action. 

Now we formulates P, E, A, S and G respectively. 

#### Enviroment E

Enviroment E is defined by the following notations：

|        Notation         | Interpretation                                               | Restrictions                                             |
| :---------------------: | ------------------------------------------------------------ | -------------------------------------------------------- |
|            n            | The chessboard size. Reversed Reversi typically has 4x4, 6x6 and 8x8 modes. | $n>=4\and n\mod2=0$                                      |
|           $t$           | The round number. **Notice that in our convention, round 0-3 exists and is played by the environment. The two agents begin to play at round 4.** | $t\in N \and t<=n^2$                                     |
|      $i = (x, y)$       | The row index and the column index.                          | $I = \{(x,y)\in N^2| 0<=x, y<n\}\\i\in I$                |
|          $C_t$          | The color that moves at round t. There are 3 possible colors, 0, 1 and -1. Two agents controls only 1 or -1. | $C=\{0,1,2\}\\C_t\in C \and C_t=2\cdot(t\mod2)-1$        |
|          $S_t$          | The state of chessboard at round t. It is a nxn matrix with color values. | $S_t\subseteq I\times C$                                 |
| $S_t(x,y)=S_{t}((x,y))$ | The color at (x, y) on the chessboard at round t.            | $S_t(x,y)\in C$                                          |
|     $ACTIONS(s, c)$     | A set of legal indexes given chessboard state and the color. | $ACTIONS(s, c)\in 2^I $                                  |
|    $RESULT(s, c,i)$     | The result chessboard after placing index i of color c on chessboard s. | $RESULT\subseteq\{(s,c,i)|i\in \\ACTIONS(s,c)\}\times S$ |
|   $TERMINAL-TEST(s)$    | Whether the                                                  |                                                          |



### Basic theorems and corollaries

#### Environment theorems

**Theorem 1.** *Monotonicity of the number of total chess pieces.* 

**Corollary 1.** *Rounds are always total pieces counts.* At any round t, we have $t=\sum_i|S_t(i)|$.

#### Performance theorems 



## Methodology

### General work flow

After having a literature review and doing a formulation, I find these problems to be vital in this project:

- Online
  - How to design an efficient and effective algorithm to **search adversarially** on the game tree? How to write it in Python with no bugs?
  - How to design a good **evaluation** function?
- Offline
  - How to **measure** the intelligence of a Reversed Reversi program?
  - How to utilize **local search** algorithms to find the best weight in evaluation function? 
  - How to generate opening books and the weights for pattern-based evaluation scheme by inverse **adversarial searching**?

Therfore, our genearl workflow 

### Assumptions



### Model design

### Model analysis

## Experiments

Experiment is very important in Computer Science Research, at least as important as it is in natural science in my opinion. 

The following experiments were performed under the following environment conditions：

- Systeminfo：AMD Ryzen 7 4800H with Radeon Graphics， Microsoft Windows 11 专业版, RAM 16G
- Python：Python 3.10.5, Pycharm 2022.1.3 (PE), numpy 1.22.4, numba 0.55.2

### Experiment 1: 探究当算法为贪心算法时不同评估函数对Agent理性程度的影响

#### Experiment principle and experiment hypothesis



#### Experiment setup and experiment steps

As we said in 4.1.1, one of the best ways to evaluate an RR agents\` rationality is to evaluate their ranking in the round robin. We need to 

##### Sub-Experiment 1.1 Baseline verification

##### Sub-Experiment 1.2 AHP combination for better 

##### Sub-Experiment 1.3 Genetic programming for local search

#### Experiment results and experiment analysis



## Conclusion and discussion



# References

[^1]: Stuartj. Russell, PeterNorvig, 诺维格, 罗素, 祝恩和殷建平, 《人工智能:一种现代的方法》, 清华大学出版社, 2013, doi: [9787302331094](https://doi.org/9787302331094).
[^2]: 彭之军, 《计算机博弈算法在黑白棋中的应用》, 现代信息科技, 卷 5, 期 17, 页 73-77+81, 2021, doi: [10.19850/j.cnki.2096-4706.2021.17.018](https://doi.org/10.19850/j.cnki.2096-4706.2021.17.018).
[^3]: C. Frankland and N. Pillay, “Evolving game playing strategies for Othello,” in *2015 IEEE Congress on Evolutionary Computation (CEC)*, Sendai, Japan, May 2015, pp. 1498–1504. doi: [10.1109/CEC.2015.7257065](https://doi.org/10.1109/CEC.2015.7257065).
[^4]: 谢国, “中国象棋机器博弈数据结构设计与搜索算法研究,” 硕士, 西安理工大学, 2008. [Online]. Available: https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD2009&filename=2008160692.nh&v=

[^5]: Orion Nebula, “黑白棋AI：局面评估+AlphaBeta剪枝预搜索,” 回答, May 09, 2018. https://zhuanlan.zhihu.com/p/35121997 (accessed Jul. 11, 2022).

[^6]: “黑白棋,” 百度百科. [https://baike.baidu.com/item/%E9%BB%91%E7%99%BD%E6%A3%8B/80689](https://baike.baidu.com/item/黑白棋/80689) (accessed Jul. 16, 2022).
[^7]: Pittner.“Reversed othello games.” http://www.tothello.com/html/reversed_othello_games.html (accessed Jul. 16, 2022).
[^8]: Teachers and assistants in SUSTech. Project-Reversed-Reversi-EN.pdf
[^9]: MacGuire.“Strategy Guide for Reversi & Reversed Reversi.” https://www.samsoft.org.uk/reversi/strategy.htm (accessed Jul. 16, 2022).
[^10]: K.-F. Lee and S. Mahajan, “The development of a world class Othello program,” *Artificial Intelligence*, vol. 43, no. 1, pp. 21–36, Apr. 1990, doi: [10.1016/0004-3702(90)90068-B](https://doi.org/10.1016/0004-3702(90)90068-B).
[^11]: Andersson.“Gunnar’s Othello page.” http://www.radagast.se/othello/ (accessed Jul. 17, 2022).



