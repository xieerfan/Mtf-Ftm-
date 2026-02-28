# Mtf/Ftm嗓音判断的小工具

## 原理

> 引自 https://gist.github.com/ypwhs/b6731d3e795ced814311573d26edd21c

使用决策树(也就是机器学习)的方法判断一段音频信号是男性还是女性,以及男女倾向百分比

判断依据为

- meanfreq: 频率平均值 (in kHz) / skew: 频谱偏度
- sd: 频率标准差 / kurt: 频谱峰度
- median: 频率中位数 (in kHz) / sp.ent: 频谱熵
- Q25: 频率第一四分位数 (in kHz) /sfm: 频谱平坦度

- Q75: 频率第三四分位数 (in kHz) / mode: 频率众数
- IQR: 频率四分位数间距 (in kHz) / centroid: 频谱质心
- meanfun: 平均基音频率 / mindom: 最小主频
- minfun: 最小基音频率 / maxdom: 最大主频
- maxfun: 最大基音频率 / dfrange: 主频范围
- meandom: 平均主频 
- modindx: 累积相邻两帧绝对基频频差除以频率范围

数据部分

> 详见:https://www.kaggle.com/datasets/primaryobjects/voicegender

在笔者写的时候,用colab拟合的数据与原文章相比深远,笔者提供的决策树权重:

<img width="3513" height="703" alt="image" src="https://github.com/user-attachments/assets/a720b658-d727-4dd6-803a-7295218ca829" />

原文章决策树权重:

<img width="2546" height="500" alt="image" src="https://github.com/user-attachments/assets/af22e28d-f945-422d-9e8c-c4077f1583e5" />

## 使用

```shell
#在安装python-3及以上的版本情况下
python -m venv venv
source venv/bin/activate.xxx #xxx为对应的系统与终端选择
pip install -r requirements.txt
```
在安装完成后,准备好测试的mp3文件

```shell
python main.py xxx.mp3
```
就会得到
<img width="1534" height="862" alt="image" src="https://github.com/user-attachments/assets/e19c1d2a-8b50-4c5c-9e32-fd7cc621e434" />
