# 基于RNN和注意力机制的双向人体姿态补全方法
##### Bi-Directional Human Pose Completion based on RNN and Attention Mechanism
Code for CADCG&amp;2021



## 1. 模型结构

<center>
    <img style="border-radius: 0.3125em;box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src=" https://z3.ax1x.com/2021/07/30/WOBkN9.png" width = "95%" alt=""/>
        </center>

## 2. 数据集

   1. [Human3.6M数据集]( https://z3.ax1x.com/2021/07/30/WOBkN9.png)
   2. [CASIA数据集](http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp)
 
## 3. 训练
运行main.py进行训练，训练策略在BiProcessor.py中进行修改
	
## 4. 实验结果
1. 模型在真实场景下的补全效果
<center>
    <img style="border-radius: 0.3125em;box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src="https://z3.ax1x.com/2021/07/30/WOtmdI.jpg" width = "65%" alt=""/>
</center>

2. 模型在Human3.6M 3D数据集中的补全效果
<center>
    <img style="border-radius: 0.3125em;box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src="https://z3.ax1x.com/2021/07/30/WO0CFA.png" width = "65%" alt=""/>
        </center>

3. 模型在Human3.6M 2D数据集中的补全效果
<center>
    <img style="border-radius: 0.3125em;box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src="https://z3.ax1x.com/2021/07/30/WO0iWt.jpg" width = "65%" alt=""/>
        </center>

4. 模型在CASIA步态数据集中的补全效果
<center>
    <img style="border-radius: 0.3125em;box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src="https://z3.ax1x.com/2021/07/30/WO0kSP.jpg" width = "65%" alt=""/>
        </center>


	