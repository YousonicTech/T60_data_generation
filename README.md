## **T60_data_generation By Hsl**

# 1. 更改path后运行
在0921_OurData_GenPT.py中直接更改三个目录
```python
parser.add_argument('--csv_file', type=str,default="/data2/hsl/0323_wav_data/add_with_zky_0316/Speech/Cas-zhongguancun/20230321T150605_test_gen_corpus_dataset_results.csv")
parser.add_argument('--dir_str', type=str,
                    default="/data2/hsl/0323_wav_data/add_with_zky_0316/Speech/Cas-zhongguancun/")
parser.add_argument('--save_dir', type=str,
                    default="/data2/hsl/0323_pt_data/add_with_zky_0316/Cas-zhongguancun/")
```
之后直接python 0921_OurData_GenPT.py 就可以直接运行，提取语谱图特征，生成对应的pt文件

# 2.生成干净语谱图
在generate_clean_pt.py中
与1中步骤一致，只不过注意将语音路径改为干净语音的路径，csv文件随便传一个就可以，然后python generate_clean_pt.py即可生成干净语谱图对应的pt文件

这一步我已经生成好了，可以直接跳过，生成的pt文件夹路径为/data2/hsl/0324_clean_pt/

# 3.将2生成的干净语谱图pt文件加入步骤1中生成的pt文件中
同样地，改好路径后，直接运行python 0927_add_clean_to_pt.py即可。

运行1-3步骤后，即可生成完整的pt文件用于训练或者测试。

# 4.附加项，多线程生成
提供了thread1.py和thread2.py,这俩功能一样，只不过路径而已，可以根据需求更改里面的路径，多线程同时生成数据。
