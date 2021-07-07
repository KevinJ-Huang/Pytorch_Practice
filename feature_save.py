import matplotlib.pyplot as plt
def feature_save(tensor,name):
    tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    inp = tensor.detach().cpu().numpy().transpose(1,2,0)
    # mean = 0.5
    # std = 0.5
    # inp = std * inp + mean
    inp = np.clip(inp,0,1)
    plt.imsave(name+'.png',inp)
