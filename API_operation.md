
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
