from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

class MyTorchDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def Torch_loader2TF_loader(torch_dataloader, batch_size, length=10e30):
  data = []
  for i, x in enumerate(torch_dataloader):
    data.append([tensor.numpy() for tensor in x])
    if i >= length*batch_size:
      break
  data = [list(x) for x in zip(*data)]
  return tf.data.Dataset.from_tensor_slices(tuple(data)).batch(batch_size)

def TF_loader2Torch_loader(tf_dataloader, batch_size, length=10e30):
  unbatch_tf_dataloader = tf_dataloader.unbatch()
  data = []
  for i, x in enumerate(unbatch_tf_dataloader):
    data.append(tuple(tensor.numpy() for tensor in x))
    if i >= length*batch_size:
      break
  return DataLoader(MyTorchDataset(data), batch_size=batch_size)