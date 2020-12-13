from Config import *
from paddle.io import Dataset
import paddle.fluid as fluid
import numpy as np
import paddle

paddle.enable_static()
datas = np.load(Config.data_path,allow_pickle=True)
data = datas['data']
#加载映射表
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()

class dataset(Dataset):
    def __init__(self, data):
        super(dataset,self).__init__()
        self.data = data

    def __getitem__(self, idx):
        poem = data[idx]
        return poem

    def __len__(self):
        return len(self.data)

train_dataset = dataset(data)
poem = paddle.static.data(name='poem', shape=[None,125], dtype='float32')
if Config.use_gpu:
    device = paddle.set_device('gpu')
    places = paddle.CUDAPlace(0)
else:
    device = paddle.set_device('cpu')
    places = paddle.CPUPlace()
paddle.disable_static(device)
train_loader = paddle.io.DataLoader(
    train_dataset,
    places=places,
    feed_list = [poem],
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=2,
    use_buffer_reader=True,
    use_shared_memory=False,
    drop_last=True,
)