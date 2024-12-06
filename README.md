如遇到依赖库版本不同引起报错，请创建虚拟环境并导入requirements.txt

利用conda创建虚拟环境方法：
1、win+R并输入cmd打开命令行

2、创建新的虚拟环境：conda create -n demo python=3.10.8
其中-n 后面跟的是环境名字，这里是demo，python后面是版本号
3、激活新环境：conda activate demo

删除虚拟环境方法：conda env remove --name demo

导入requirements.txt方法：
1、下载requirements.txt文件
2、打开命令行并切换到requirements.txt文件目录下
3、pip install -r requirements.txt 安装文件
安装完成后可在命令行中使用pip list命令查看导入的库
