import matplotlib.pyplot as plt
from DataLoader import *
from Config import *
from Network import *
from paddle.io import Dataset
import paddle.fluid as fluid
import numpy as np
import paddle
loss_ = []

def train():
    model = Peom(
        len(word2ix),
        embedding_dim=Config.embedding_dim,
        hidden_dim=Config.hidden_dim
    )
    # state_dict = paddle.load(Config.model_path)
    # model.set_state_dict(state_dict)
    optim = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=Config.lr)
    lossf = paddle.nn.CrossEntropyLoss()
    for epoch in range(Config.epoch):
        for li, data in enumerate(train_loader()):
            optim.clear_grad()
            data = data[0]
            # data = paddle.transpose(data,(1,0))
            x = paddle.to_tensor(data[:, :-1])
            y = paddle.to_tensor(data[:, 1:], dtype='int64')
            y = paddle.reshape(y, [-1])
            y = paddle.to_tensor(y, dtype='int64')
            output, hidden = model(x)
            loss = lossf(output, y)
            loss.backward()
            optim.step()
            loss_.append(loss.numpy()[0])

            if li % Config.plot_every == 0:
                print('Epoch ID={0}\t Batch ID={1}\t Loss={2}'.format(epoch, li, loss.numpy()[0]))

                results = list(Config.start_words)
                start_words_len = len(Config.start_words)
                # 第一个词语是<START>
                input = paddle.to_tensor([word2ix['<START>']])
                input = paddle.reshape(input, [1, 1])
                hidden = None

                # 若有风格前缀，则先用风格前缀生成hidden
                if Config.prefix_words:
                    # 第一个input是<START>，后面就是prefix中的汉字
                    # 第一个hidden是None，后面就是前面生成的hidden
                    for word in Config.prefix_words:
                        output, hidden = model(input, hidden)
                        input = paddle.to_tensor([word2ix[word]])
                        input = paddle.reshape(input, [1, 1])

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
                        input = paddle.reshape(input, [1, 1])
                    # 否则将output作为下一个input进行
                    else:
                        # print(output.data[0].topk(1))
                        _, top_index = paddle.fluid.layers.topk(output[0], k=1)
                        top_index = top_index.numpy()[0]
                        w = ix2word[top_index]
                        results.append(w)
                        input = paddle.to_tensor([top_index])
                        input = paddle.reshape(input, [1, 1])
                    if w == '<EOP>':
                        del results[-1]
                        break
                results = ''.join(results)
                print(results)
        paddle.save(model.state_dict(), Config.model_prefix)


train()
plt.figure(figsize=(15, 6))
x = np.arange(len(loss_))
plt.title('Loss During Training')
plt.xlabel('Number of Batch')
plt.plot(x,np.array(loss_))
plt.savefig('work/Loss During Training.png')
plt.show()
