# AI Project2: Development of an efficient CARP

#  solving agent based on a variant of MAENS with 

# limited computational budget 

# AI 项目2: 算时受限情形下的基于带扩展邻居搜索的文

# 化基因演化变种方法的容

# 量约束弧路径问题的求解智能体的开发

叶璨铭，[12011404@mail.sustech.edu.cn](mailto:12011404@mail.sustech.edu.cn)

## Introduction

## Methodology

### Software architecture

###  Model design

```pseudocode
Function Budget-Limited-MAENS(pop_size, budget, pr_ls)
	will return an evolved population 
	inputs: 
		pop_size: population size as the evolution algorithm
		budget: the maximum number of function evaluations(MFE)
		pr_ls: probability for local search
    
    Q = new Population()
    while evaluated <= budget do
    	Apply MSA inspired selection operator to select pop_size/2 pairs of parents
    	Save the common trips of each pair of parents, keep the different part of each pairs, i.e., a set of edges that contains different trips, as P ∗
    	
    	
    end while
    
		
```



## Experiment result and analysis

## Conclusion

# References

[^1]: K. Tang, Y. Mei, and X. Yao, “Memetic algorithm with extended neighborhood search for capacitated arc routing problems,” *IEEE Transactions on Evolutionary Computation*, vol. 13, no. 5, pp. 1151–1166, 2009, doi: [10.1109/TEVC.2009.2023449](https://doi.org/10.1109/TEVC.2009.2023449).
[^2]: M. Liu and T. Ray, “Efficient Solution of Capacitated Arc Routing Problems with a Limited Computational Budget,” in *AI 2012: Advances in Artificial Intelligence*, vol. 7691, M. Thielscher and D. Zhang, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012, pp. 791–802. doi: [10.1007/978-3-642-35101-3_67](https://doi.org/10.1007/978-3-642-35101-3_67).
[^3]: “The fleet size and mix problem for capacitated arc routing.” https://reader.elsevier.com/reader/sd/pii/0377221785902528?token=C8A1FEC554B5CD4789BFFE151D083E76143F238EBDB7B118CBC2F1F33560B56B05048E5D98465B82E41D2882FE5746D1&originRegion=us-east-1&originCreation=20220805155926 (accessed Aug. 06, 2022).
[^4]: Devin Yang, “Python-argparse-命令行与参数解析,” 相信简单的力量, Mar. 09, 2018. https://zhuanlan.zhihu.com/p/34395749 (accessed Aug. 05, 2022).
[^5]: “floyd算法(C语言 邻接表)_cjavacjavacjava的博客-CSDN博客_floyd算法邻接表.” https://blog.csdn.net/cjavacjavacjava/article/details/73949087 (accessed Aug. 05, 2022).
[^6]: 赵耀. CARP问题图解.ppt
[^7]: C. Huang, “Heuristics for CARP,” p. 17.

