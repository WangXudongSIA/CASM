import torch
import torch.optim
from model import *
from data import MyDataset
from data import Myval_Dataset
from utils import weights_init
from torch.utils.data import DataLoader
import hiddenlayer as hl
from utils import FeatureLoss
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def train(train_orig_images_path, train_hazy_images_path,val_orig_images_path, val_hazy_images_path, train_batch_size, val_batch_size, epochs):
    train_dataset = MyDataset(train_orig_images_path, train_hazy_images_path)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = Myval_Dataset(val_orig_images_path, val_hazy_images_path)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    dehaze_net = MdehazeNet()
    dehaze_net.apply(weights_init)
    # dehaze_net = torch.load('saved_models/TBN.pt')
    dehaze_net = dehaze_net.cuda()
    criterion = FeatureLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dehaze_net.parameters()),
                                 lr=1e-3, weight_decay=1e-4)

    history = hl.History()
    canvas = hl.Canvas()
    valsize = len(val_loader)
    trainsize = len(train_loader)
    zz = 0
    ii = 0
    for epoch in range(epochs):
        start.record()
        epoch_train_loss = 0
        epoch_evaluate_loss = 0
        zz_train_loss = 0

        dehaze_net.eval()
        for iteration, (img_orig, img_haze) in enumerate(val_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()
            with torch.no_grad():
                clean_image = dehaze_net(img_haze)
            loss = criterion(clean_image, img_orig)
            epoch_evaluate_loss += loss.item()

        dehaze_net.train()
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            optimizer.zero_grad()
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()
            clean_image = dehaze_net(img_haze)
            loss = criterion(clean_image, img_orig)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), 0.1)
            epoch_train_loss += loss.item()
            zz_train_loss += loss.item()
            zz = zz + 1
            if zz % 100 == 0:
                ii = ii + 1
                history.log(ii, train_loss=zz_train_loss / zz, val_loss = epoch_evaluate_loss / valsize)
                with canvas:
                    canvas.draw_plot([history["train_loss"], history["val_loss"]])
                zz_train_loss = 0
                zz = 0
                torch.save(dehaze_net, 'saved_models/TBN.pt')
        end.record()
        torch.cuda.synchronize()
        print('EPOCH : %03d, Validation_LOSS: %2.3f, Train_LOSS: %2.3f' % (epoch+1, epoch_evaluate_loss / valsize, epoch_train_loss / trainsize))
        torch.save(dehaze_net, 'saved_models/TBN.pt')

if __name__ == '__main__':

    train_orig_images_path = './train/gt/'
    train_hazy_images_path = './train/hazy/'
    val_orig_images_path = './validation_data/gt/'
    val_hazy_images_path = './validation_data/hazy/'

    train_batch_size = 2
    val_batch_size = 32
    epochs = 50
    train(train_orig_images_path, train_hazy_images_path, val_orig_images_path, val_hazy_images_path, train_batch_size, val_batch_size, epochs)
