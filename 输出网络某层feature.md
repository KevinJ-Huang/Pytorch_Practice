#### 如果某个网络是net,输出其中deconv10层的feature，使用以下代码：
```python
            if  cur_iter%1000 == 0:
                activation = {}
                def get_activation(name):
                    def hook(model, input, output):
                        # ????feature????????????? detach??
                        activation[name] = output.detach()

                    return hook
                model = self.net
                model.deconv10.register_forward_hook(get_activation('deconv10'))
                output = model(lr_imgs)

                print('res:', activation['deconv10'].mean())
                count+=1
```
