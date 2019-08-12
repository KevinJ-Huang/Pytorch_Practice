
#### 1.索引部分参数
```python
data = tensor.narrow(dim, index, size) 
–表示取出tensor中第dim维上索引从index开始到index+size-1的所有元素存放在data中 
```

#### 2.pytorch numpy转tensor,tensor转numpy
```python
a = np.ones(5)
torch.from_numpy(a)

a = torch.FloatTensor(3,3)
print a.numpy()
```

#### 3.pytorch扩展维度

```python
sig = torch.from_numpy(sig).float().unsqueeze(0)
```

#### 4.如果保存的模型是在多个GPU上并行计算的DataParallel类型。在读取模型时，需要先取出DataParallel的一般module模型
```python
    保存的模型类型为DataParallel
    network_parallel = DataParallel(network, gpu_ids)
    torch.save(network_parallel, "modelpara_weights_path")
    # 读取的模型类型为DataParallel, 现将其读取到CPU上
    model = torch.load("modelpara_weights_path", map_location=lambda storage, loc: storage)
    # 取出DataParallel中的一般module类型
    model_module = model.module
```
