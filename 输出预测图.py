import os
import argparse
import scipy.misc as misc
import torch
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
from data import PreSuDataset
from utils import tensor_to_img, calc_metric_per_img, Config
from net import UNet

from skimage.io import imsave
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

parser = argparse.ArgumentParser(description='Predict with Deep Guided Filtering Networks')
parser.add_argument('--task',        type=str, default='',                  help='TASK')
parser.add_argument('--img_path',    type=str, default='/userhome/dped/validation/input/',               help='IMG_PATH')
parser.add_argument('--img_list',    type=str, default='/userhome/MYDGF/run/dataset.txt',          help='IMG_LIST')
parser.add_argument('--save_folder', type=str, default='/userhome/MYDGF/run/output/',           help='SAVE_FOLDER')
parser.add_argument('--gpu',         type=int, default=0,                          help='GPU')
parser.add_argument('--gray',         default=False, action='store_true', help='GPU')
parser.add_argument('--model_path', type=str, default='/userhome/MYDGF/checkpoints_dped/snapshots/', help='MODEL_FOLDER')
parser.add_argument('--model_id', type=str, default='net_epoch_11.pth', help='MODEL_ID')
args = parser.parse_args()


# Save Folder
if not os.path.isdir(args.save_folder):
    os.makedirs(args.save_folder)

model = UNet()


model_path = os.path.join(args.model_path, args.task, args.model_id)
print("load...",model_path)

model.load_state_dict(torch.load(model_path))

# data set
input_path = args.img_path
input_images = os.listdir(input_path)

# GPU
if args.gpu >= 0:
    with torch.cuda.device(args.gpu):
        model.cuda()

# test
i_bar = tqdm(total=len(input_images), desc='#Images')
for idx, imgs in enumerate(input_images):
    # if idx == 0:
    #     idx+=1
    image_phone = misc.imread(input_path + imgs) / 255.0
    name = os.path.basename(imgs)
    image_phone = np.transpose(image_phone, [2, 0, 1])
    images_iphone = torch.from_numpy(image_phone).float()

    hr_x = images_iphone.unsqueeze(0)
    if args.gpu >= 0:
        with torch.cuda.device(args.gpu):
            hr_x = hr_x.cuda()
    imgs = model(Variable(hr_x)).data.cpu()

    for img in imgs:
        img = tensor_to_img(img, transpose=True)
        if args.gray:
            img = img.mean(axis=2).astype(img.dtype)
        imsave(os.path.join(args.save_folder, name), img)

    i_bar.update()
