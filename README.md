# tinyImageNet
ImageNet数据集太大分出来了MiniImageNet，觉得MiniImageNet太大了又分出了TinyImageNet
这个是基于MiniImageNet数据集的，MiniImageNet文件夹下刚开始只有一个压缩包，三个csv文件（train.csv，test.csv，val.csv），以及imagenet_class_index.json。首先第一步把压缩包解压，出现images文件夹
但是训练集、测试集、验证集的类别不均匀，restructure_csv.py可以生成均匀的训练集与测试集csv文件
经历过上述步骤后，creat_tiny.py可以创建一个更小型的图片集
creat_pickle.py则可以将处理图片集之后的结果保存下来
