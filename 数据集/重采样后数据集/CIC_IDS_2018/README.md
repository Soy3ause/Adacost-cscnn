对CIC_IDS_2017进行如下表的重采样操作：


类别名	         重采样前数量	重采样后数量
Benign	    13390249	1330000
DoS/DDoS	    1918233	      350000
Brute Force	    380943	      75500
Botnet	    286191	      66000
Infilteration   160639	      65500
Web Attack	    928	      65000



注：重采样后将标签名为“BENIGN"的数据保留，作为良性流量；将其余标签名更改为”Malicious"，作为恶意攻击流量，由此得到重采样后的二分类数据集
