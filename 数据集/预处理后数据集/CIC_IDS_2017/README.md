CIC_IDS_2017数据预处理经过了以下步骤：

1、数据表拼接：将官网下载得到的多个日期的数据表拼接为一个数据表

2、处理缺失值和异常值：由于原始数据集中存在inf和null值，将inf替换为null后丢弃所有null值

3、标签处理：
对于二分类：
将标签名为“BENIGN"的数据保留，作为良性流量；将其余标签名更改为”Malicious"，作为恶意攻击流量

对于多分类按照下表进行处理：

| 合并后标签名   | 原始标签名                                                             |
| ------------ |:---------------------------------------------------------------------|
|  Benign      | Benign                                                               |
|  DoS/DDoS    | Heartbleed,DDoS,DoS Hulk,DoS GoldenEye,DoS Slowloris,DoS Slowhttptest|
|  PortScan    | PortScan                                                             |
|  Brute Force | FTP-Patator,SSH-Patator                                              |
|  Web Attack  | Web Attack - Burte Force,Web Attack - XSS,Web Attack - SQL Injection |
|  Botnet      | Bot                                                                  |

4、特征筛选：
| N°    | 特征名                              |  保留  |   修改   |   丢弃   |   备注      |
| ----- |:----------------------------------:| -----:| --------:| -------:|-----------:|
| 0     | destination\_port                  |   x   |          |         |            |
| 1     | flow\_duration                     |       |          |    x    |   高相关    |
| 2-3   | total\_fwd/bwd\_packet             |       |          |    x    |   高相关    |
| 4-5   | total\_length\_of\_fwd/bwd\_packet |       |          |    x    |   高相关    |
| 6-7   | fwd\_packet\_length\_max/mean      |       |          |    x    |   高相关    |
| 8-9   | fwd\_packet\_length\_min/std       |   x   |          |         |            |
| 10-11 | bwd\_packet\_length\_max/mean      |       |          |    x    |   高相关    |
| 12-13 | bwd\_packet\_length\_min/std       |   x   |          |         |            |
| 14    | flow\_bytes/s                      |       |     x    |         | NaN/Infinity|
| 15    | flow\_packet/s                     |       |     x    |         | NaN/Infinity|
| 16-19 | flow\_iat\_mean/std/max/min        |   x   |          |         |            |
| 20-24 | fwd\_iat\_total/mean/std/max/min   |   x   |          |         |            |
| 25-29 | bwd\_iat\_total/mean/std/max/min   |   x   |          |         |            |
| 30    | fwd\_psh\_flag                     |       |          |    x    |   高相关    |
| 31    | bwd\_psh\_flag                     |       |          |    x    |  数据无差异  |
| 32    | fwd\_urg\_flag                     |       |          |    x    |  数据无差异  |
| 33    | bwd\_urg\_flag                     |       |          |    x    |  数据无差异  |
| 34-35 | fwd/bwd\_header\_length            |   x   |          |         |            |
| 36-37 | fwd/bwd\_packet/s                  |   x   |          |         |            |
| 38-39 | packet\_length\_max/mean           |   x   |          |         |            |
| 40-42 | packet\_length\_min/std/variance   |       |          |    x    |   高相关    |
| 43    | fin\_flag\_count                   |   x   |          |         |            |
| 44    | syn\_flag\_count                   |   x   |          |         |            |
| 45    | rst\_flag\_count                   |       |          |    x    |   高相关    |
| 46    | psh\_flag\_count                   |   x   |          |         |            |
| 47    | ack\_flag\_count                   |   x   |          |         |            |
| 48    | urg\_flag\_count                   |   x   |          |         |            |
| 49    | cwe\_flag\_count                   |       |          |    x    |  数据无差异  |
| 50    | ece\_flag\_count                   |   x   |          |         |            |
| 51    | down/up\_ratio                     |   x   |          |         |            |
| 52    | average\_packet\_size              |   x   |          |         |            |
| 53-54 | avg\_fwd/bwd\_segment size         |   x   |          |         |            |
| 55    | fwd\_header\_length.1              |       |          |    x    |   重复特征  |
| 56    | fwd\_avg\_bytes/bulk               |       |          |    x    |  数据无差异  |
| 57    | fwd\_avg\_packet/bulk              |       |          |    x    |  数据无差异  |
| 58    | fwd\_avg\_bulk rate                |       |          |    x    |  数据无差异  |
| 59    | bwd\_avg\_bytes/bulk               |       |          |    x    |  数据无差异  |
| 60    | bwd\_avg\_packet/bulk              |       |          |    x    |  数据无差异  |
| 61    | bwd\_avg\_bulk rate                |       |          |    x    |  数据无差异  |
| 62-63 | subflow\_fwd/bwd\_packets          |       |          |    x    |   高相关    | 
| 64-65 | subflow\_fwd/bwd\_bytes            |   x   |          |         |            |
| 66    | init\_win\_bytes\_forward          |   x   |          |         |            |
| 67    | init\_win\_bytes\_backward         |   x   |          |         |            |
| 68    | act\_data\_pkt\_fwd                |   x   |          |         |            |
| 69    | min\_seg\_size\_forward            |   x   |          |         |            |
| 70-73 | active\_mean/std/max/min           |   x   |          |         |            |
| 74-75 | idle\_mean/max                     |       |          |    x    |   高相关    |
| 76-77 | idle\_std/min                      |   x   |          |         |            |
| 78    | label                              |       |          |    x    |    标签    |

5、标准化和归一化：使用分位数转换方法对数据进行标准化和归一化