CIC_IDS_2018数据预处理经过了以下步骤：

1、数据表拼接：将官网下载得到的多个日期的数据表拼接为一个数据表

2、处理缺失值和异常值：由于原始数据集中存在inf和null值，将inf替换为null后丢弃所有null值

3、标签处理：
对于二分类：
将标签名为“BENIGN"的数据保留，作为良性流量；将其余标签名更改为”Malicious"，作为恶意攻击流量

对于多分类按照下表进行合并：

| 合并后标签名    | 原始标签名                                                                                                                                          |
| ------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------|
|  Benign       | Benign                                                                                                                                            |
|  DoS/DDoS     | DDOS attack-HOIC,DDoS attacks-LOIC-HTTP,DoS attacks-Hulk,DoS attacks-SlowHTTPTest,DoS attacks-GoldenEye,DoS attacks-Slowloris,DDOS attack-LOIC-UDP|
|  Infilteration| PortScan                                                                                                                                          |
|  Brute Force  | FTP-BruteForce,SSH-Bruteforce                                                                                                                     |
|  Web Attack   | Brute Force -Web,Brute Force -XSS,SQL Injection                                                                                                   |
|  Botnet       | Bot                                                                                                                                               |

4、特征筛选：
N°      特征名                                                        保留       修改     丢弃                备注
---------------------------------------------------------------------
0       Dst Port                          x                        
1       Protocol                                       x       高相关
2       Flow Duration                                  x       高相关
3       Timestamp                                      x       冗余
4       Tot Fwd Pkts                                   x       高相关
5       Tot Bwd Pkts                                   x       高相关
6       TotLen Fwd Pkts                                x       高相关
7       TotLen Bwd Pkts                                x       高相关
8       Fwd Pkt Len Max                                x       高相关
9       Fwd Pkt Len Min                   x                        
10      Fwd Pkt Len Mean                               x       高相关
11      Fwd Pkt Len Std                   x                        
12      Bwd Pkt Len Max                                x       高相关
13      Bwd Pkt Len Min                   x                        
14      Bwd Pkt Len Mean                               x       高相关
15      Bwd Pkt Len Std                   x                        
16      Flow Byts/s                               x            检查NaN        
17      Flow Pkts/s                               x            检查NaN        
18      Flow IAT Mean                     x                        
19      Flow IAT Std                      x                        
20      Flow IAT Max                      x                        
21      Flow IAT Min                      x                        
22      Fwd IAT Tot                       x                        
23      Fwd IAT Mean                      x                        
24      Fwd IAT Std                       x                        
25      Fwd IAT Max                       x                        
26      Fwd IAT Min                       x                        
27      Bwd IAT Tot                       x                        
28      Bwd IAT Mean                      x                        
29      Bwd IAT Std                       x                        
30      Bwd IAT Max                       x                        
31      Bwd IAT Min                       x                        
32      Fwd PSH Flags                                   x       高相关
33      Bwd PSH Flags                                   x       数据无差异
34      Fwd URG Flags                                   x       数据无差异
35      Bwd URG Flags                                   x       数据无差异
36      Fwd Header Len                    x                        
37      Bwd Header Len                    x                        
38      Fwd Pkts/s                                 x           检查NaN        
39      Bwd Pkts/s                                 x           检查NaN        
40      Pkt Len Min                       x                        
41      Pkt Len Max                       x                        
42      Pkt Len Mean                      x                        
43      Pkt Len Std                                     x       高相关
44      Pkt Len Var                                     x       高相关
45      FIN Flag Cnt                      x                        
46      SYN Flag Cnt                      x                        
47      RST Flag Cnt                                    x       高相关
48      PSH Flag Cnt                      x                        
49      ACK Flag Cnt                      x                        
50      URG Flag Cnt                      x                        
51      CWE Flag Count                                  x       数据无差异
52      ECE Flag Cnt                      x                        
53      Down/Up Ratio                     x                        
54      Pkt Size Avg                      x                        
55      Fwd Seg Size Avg                  x                        
56      Bwd Seg Size Avg                  x                        
57-62   Fwd/Bwd Byts/b, Pkts/b, Blk Rate Avg            x       高相关
63      Subflow Fwd Pkts                                x       高相关
64      Subflow Fwd Byts                  x                        
65      Subflow Bwd Pkts                                x       高相关
66      Subflow Bwd Byts                  x                        
67      Init Fwd Win Byts                 x                        
68      Init Bwd Win Byts                 x                        
69      Fwd Act Data Pkts                 x                        
70      Fwd Seg Size Min                  x                        
71-74   Active Mean/Std/Max/Min           x                        
75      Idle Mean                                       x       高相关
76      Idle Std                          x                        
77      Idle Max                                        x       高相关
78      Idle Min                          x                        
79      Label                             x                    标签

5、标准化和归一化：使用分位数转换方法对数据进行标准化和归一化

