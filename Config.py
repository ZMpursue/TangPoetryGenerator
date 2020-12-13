from paddle.io import Dataset
import paddle.fluid as fluid
import numpy as np
import paddle
class Config(object):
    num_layers = 3                                      # LSTM层数
    data_path = './tang_poem.npz'                    # 诗歌的文本文件存放路径
    lr = 1e-3                                           # 学习率
    use_gpu = True                                      # 是否使用GPU
    epoch = 20
    batch_size = 4                                      # mini-batch大小
    maxlen = 125                                        # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 1000                                   # 隔batch 可视化一次
    max_gen_len = 200                                   # 生成诗歌最长长度
    model_path = "./checkpoints/model.params.50"     # 预训练模型路径
    prefix_words = '欲穷千里目，更上一层楼'                 # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '老夫聊发少年狂，'                       # 诗歌开始
    model_prefix = './checkpoints/model.params'      # 模型保存路径
    embedding_dim = 256                                 # 词向量维度
    hidden_dim = 512                                    # LSTM hidden层维度