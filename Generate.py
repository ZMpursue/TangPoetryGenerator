from DataLoader import *
from Config import *
from paddle.io import Dataset
import paddle.fluid as fluid
import numpy as np
import paddle

# 给定首句生成诗歌
def generate(model, start_words, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = paddle.to_tensor([word2ix['<START>']])
    input = paddle.reshape(input,[1,1])
    hidden = None

    # 若有风格前缀，则先用风格前缀生成hidden
    if prefix_words:
        # 第一个input是<START>，后面就是prefix中的汉字
        # 第一个hidden是None，后面就是前面生成的hidden
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = paddle.to_tensor([word2ix[word]])
            input = paddle.reshape(input,[1,1])

    # 开始真正生成诗句，如果没有使用风格前缀，则hidden = None，input = <START>
    # 否则，input就是风格前缀的最后一个词语，hidden也是生成出来的
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        # print(output.shape)
         # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到
        # 最后的hidden
        if i < start_words_len:
            w = results[i]
            input = paddle.to_tensor([word2ix[w]])
            input = paddle.reshape(input,[1,1])
        # 否则将output作为下一个input进行
        else:
            # print(output.data[0].topk(1))
            _,top_index = paddle.fluid.layers.topk(output[0],k=1)
            top_index = top_index.numpy()[0]
            w = ix2word[top_index]
            results.append(w)
            input = paddle.to_tensor([top_index])
            input = paddle.reshape(input,[1,1])
        if w == '<EOP>':
            del results[-1]
            break
    results = ''.join(results)
    return results


# 生成藏头诗
def gen_acrostic(model, start_words, prefix_words=None):
    result = []
    start_words_len = len(start_words)
    input = paddle.to_tensor([word2ix['<START>']])
    input = paddle.reshape(input,[1,1])
    # 指示已经生成了几句藏头诗
    index = 0
    pre_word = '<START>'
    hidden = None

    # 存在风格前缀，则生成hidden
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = paddle.to_tensor([word2ix[word]])
            input = paddle.reshape(input,[1,1])

    # 开始生成诗句
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        _,top_index = paddle.fluid.layers.topk(output[0],k=1)
        top_index = top_index.numpy()[0]
        w = ix2word[top_index]
        # 说明上个字是句末
        if pre_word in {'。', '，', '?', '！', '<START>'}:
            if index == start_words_len:
                break
            else:
                w = start_words[index]
                index += 1
                # print(w,word2ix[w])
                input = paddle.to_tensor([word2ix[w]])
                input = paddle.reshape(input,[1,1])
        else:
            input = paddle.to_tensor([top_index])
            input = paddle.reshape(input,[1,1])
        result.append(w)
        pre_word = w
    result = ''.join(result)
    return result

#读取训练好的模型
model = Peom(
        len(word2ix),
        embedding_dim = Config.embedding_dim,
        hidden_dim = Config.hidden_dim
    )
state_dict = paddle.load('work/checkpoints/model.params.50')
model.set_state_dict(state_dict)
print('[*]模式一：首句续写唐诗：')
#生成首句续写唐诗（prefix_words为生成意境的诗句）
print(generate(model, start_words="春江潮水连海平，", prefix_words="滚滚长江东逝水，浪花淘尽英雄。"))
print('[*]模式二：藏头诗：')
#生成藏头诗
print(gen_acrostic(model, start_words="夜月一帘幽梦春风十里柔情", prefix_words="落花人独立，微雨燕双飞。"))