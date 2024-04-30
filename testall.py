import torch
import torch.optim
import numpy as np
from PIL import Image
from utils import PSNR
from utils import SSIM
from torchvision import transforms
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def dehaze_image(image_name, original_name):
    data_hazy = Image.open(image_name)
    data_hazy = np.array(data_hazy) / 255.0
    original_img = Image.open(original_name)
    original_img = np.array(original_img) / 255.0
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.unsqueeze(0)

    dehaze_net = torch.load('saved_models/TBN.pt', map_location=torch.device('cpu')).cuda()
    dehaze_net.eval()

    start.record()
    torch.cuda.synchronize()
    with torch.no_grad():
        clean_image = dehaze_net(data_hazy.cuda())
    end.record()

    torch.cuda.synchronize()
    mtime = start.elapsed_time(end)

    clean_image = clean_image.cpu().squeeze()
    toPIL = transforms.ToPILImage()
    img = clean_image
    image = toPIL(img)
    image.save(image_name.replace('hazy', 'mresult'))
    clean_image = clean_image.detach().numpy()
    clean_image = np.swapaxes(clean_image, 0, 1)
    clean_image = np.swapaxes(clean_image, 1, 2)
    mPSNR = PSNR(original_img, clean_image)
    print("PSNR: %3.2f" % mPSNR)
    mSSIM = SSIM(original_img, clean_image)
    print("SSIM: %3.3f" % mSSIM)
    print('Time: %3.2f ms' % mtime)

    return mPSNR, mSSIM, mtime


if __name__ == '__main__':

    add = './test0/hazy/'
    ind = 0
    addlist = sorted(os.listdir(add))
    n = len(addlist)
    mPSNR_array = [0 for x in range(n)]
    mSSIM_array = [0 for x in range(n)]
    mMSE_array = [0 for x in range(n)]
    mtime_array = [0 for x in range(n)]


    for filename in addlist:
        ind = ind + 1
        print("********* %d th test result *********" % ind)
        img_name = add + filename
        original_name = add.replace('hazy', 'gt') + filename
        mPSNR, mSSIM, mtime = dehaze_image(img_name, original_name)
        mPSNR_array[ind - 1] = mPSNR
        mSSIM_array[ind - 1] = mSSIM
        mtime_array[ind - 1] = mtime

    mPSNR1 = sum(mPSNR_array) / len(mPSNR_array)
    mSSIM1 = sum(mSSIM_array) / len(mSSIM_array)
    del mtime_array[0]
    mtime1 = sum(mtime_array) / len(mtime_array)

    print("---------*******-All Result-*******--------")
    print('Time: %3.2f ms' % mtime1)
    print("PSNR: %3.2f" % mPSNR1)
    print("SSIM: %3.3f" % mSSIM1)
