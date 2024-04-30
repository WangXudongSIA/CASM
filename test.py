import torch
import torch.optim
from PIL import Image
from torchvision import transforms
from pylab import *
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def dehaze_image(image_name):

    data_hazy = Image.open(image_name)
    o_image = data_hazy
    data_hazy = np.array(data_hazy) / 255.0

    data_hazy = torch.from_numpy(data_hazy).float()

    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.unsqueeze(0)

    dehaze_net1 = torch.load('saved_models/TBN.pt', map_location=torch.device('cpu')).cuda()

    dehaze_net = torch.load('saved_models/TBN0.pt', map_location=torch.device('cpu')).cuda()

    dehaze_net.eval()
    dehaze_net1.eval()

    start.record()
    torch.cuda.synchronize()
    with torch.no_grad():
        clean_image1 = dehaze_net1(data_hazy.cuda())
    end.record()
    torch.cuda.synchronize()
    clean_image1 = clean_image1.cpu().squeeze()

    # toPIL = transforms.ToPILImage()
    # img = clean_image1
    # image = toPIL(img)
    # image.save('Supervised_Dehazing.png')

    clean_image1 = clean_image1.detach().numpy()
    clean_image1 = np.swapaxes(clean_image1, 0, 1)
    clean_image1 = np.swapaxes(clean_image1, 1, 2)

    start.record()
    torch.cuda.synchronize()
    with torch.no_grad():
        clean_image = dehaze_net(data_hazy.cuda())
    end.record()
    torch.cuda.synchronize()
    clean_image = clean_image.cpu().squeeze()

    toPIL = transforms.ToPILImage()
    img = clean_image
    image = toPIL(img)
    image.save('Semi_Supervised_Dehazing.png')

    clean_image = clean_image.detach().numpy()
    clean_image = np.swapaxes(clean_image, 0, 1)
    clean_image = np.swapaxes(clean_image, 1, 2)

    plt.subplot(1, 3, 1)
    plt.imshow(o_image)
    plt.axis('off')
    plt.title('Hazy Image', fontdict={'fontsize': 8})

    plt.subplot(1, 3, 2)
    plt.imshow(clean_image1)
    plt.axis('off')
    plt.title('Supervised Learning', fontdict={'fontsize': 8})

    plt.subplot(1, 3, 3)
    plt.imshow(clean_image)
    plt.axis('off')
    plt.title('Semi-Supervised Learning', fontdict={'fontsize': 8})
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    img_name = ('./test_images/test.png')
    dehaze_image(img_name)





