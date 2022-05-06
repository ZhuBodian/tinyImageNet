# tinyImageNet
ImageNet数据集太大分出来了MiniImageNet，觉得MiniImageNet太大了又分出了TinyImageNet，MiniImageNet含有100个类，每个类600张图片，总共60000张图片，如果把图片剪裁为224×224像素的，3通道，采用float64，总共需要内存60000*224*224*3*8/1024/1024/1024=67G。

这个是基于MiniImageNet数据集的，MiniImageNet文件夹下刚开始只有一个压缩包，三个csv文件（train.csv，test.csv，val.csv），以及imagenet_class_index.json。
csv文件中的标签是数字字母混合类型的，json文件则给出了其对应的纯数字标签与纯文本标签（有意义的单词）。

首先第一步把压缩包解压，出现images文件夹，但是训练集、测试集、验证集的类别不均匀。

creat_tiny.py可以生成一个新的tinyImageNet文件夹，并生成均匀的训练集与测试集csv文件（这里csv文件中的标签则改为了单词标签）。

注意MiniImageNet的纯数字标签是0~99的连续标签，给creat_tiny.py的extraction order是一个数字列表（数字对应的单词标签查看imagenet_class_index.json）

creat_pickle.py则可以将处理图片集之后的结果保存下来
