import cv2
import os
import argparse
import numpy as np
import torch
import lpips



def main(args):
    loss_fn_alex = lpips.LPIPS(net='alex')
    LPIPS = []
    # LPIPS_total = 0
    i = 0
    folder_list = sorted(os.listdir(args.input))
    # for folder in folder_list:
    image_list = sorted(os.listdir(os.path.join(args.input)))
    for file in image_list:
        # i+=1
        # if i%9==0:

        input = os.path.join(args.input,file)
        folder = file.split('_')[0]
        filename = file.split('_')[1]

        target = os.path.join(args.target,folder,filename)
        input_img = cv2.imread(input,-1)
        target_img = cv2.imread(target,-1)
        if input_img is None or target_img is None:
            continue
        input_img = torch.from_numpy(np.ascontiguousarray(np.transpose(input_img, (2, 0, 1)))).float().unsqueeze(0)
        target_img = torch.from_numpy(np.ascontiguousarray(np.transpose(target_img, (2, 0, 1)))).float().unsqueeze(0)
        lpips_score = loss_fn_alex(input_img,target_img)
        # psnr,ssim = 0,0
        print("LPIPS:%.4f"%(lpips_score))


        with open(args.input+'LPISP.txt','a') as f:
            f.write(' %.4f \n' % lpips_score)
        # LPIPS_total+=lpips_score
        # i+=0
        LPIPS.append(lpips_score)
        # SSIM.append(ssim)
        # with open('name.txt','a') as f:
    #     f.write(' %s\n'%file)

    print("AVG LPIPS:%.4f"%(np.mean(np.array(LPIPS))))
    # print("AVG LPIPS:%.4f" % LPIPS_total/(i*1.0))

# holopix50_0202V2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    pos = 'Ours'
    parser.add_argument('--pos',default=pos,type=str, help='pos')
    parser.add_argument('--input', default='/data/1760921465/NTIRE2021/Deblocking/'+pos,type=str, help='input images path')
    parser.add_argument('--target', default='/data/1760921465/NTIRE2021/val_sharp/',type=str, help='output images path')
    args = parser.parse_args()

    main(args)
