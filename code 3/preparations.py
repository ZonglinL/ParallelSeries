from config import get_config
from model import *
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Loaders import *
from tools import StandardScaler

import math

from tqdm import tqdm
import csv
import os



config = get_config()
data_name = config.data_name


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(config.data_dir) as f:
    data_list = csv.reader(f)
    data = []
    for row in data_list:
        data.append(row)
    f.close()

data = torch.tensor(np.array(data[1:]).astype(np.float))
data = data.type(torch.float)
train_len = math.floor(data.shape[0]*config.train_share)
test_len = math.floor(data.shape[0]*0.8)
if config.scale:
    mean = torch.mean(data[:train_len,:],dim = 0)
    std = torch.std(data[:train_len,:],dim = 0)
    data = (data - mean) / std



def save_model(config, model, epoch_index):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(config.output_dir, "epoch%s_checkpoint.bin" % epoch_index)
    torch.save(model_to_save.state_dict(), model_checkpoint)


def train_epoch(model, optimizer,  train_dl, t0=96, dec_len = 24, to_predict=24, noise = 0):
    criterion = config.criterion
    model.train()
    train_loss = 0
    train_KL = 0
    n = 0
    for step, (x, y, feature, x_dec, y_dec, feature_dec, attention_masks, dec_mask) in enumerate(
            train_dl):  # using dataloader, a library in pytorch. It helps
        # carry out batching traing
        # google torch.utils.data.DataLoader
        x = x.to(device)
        y = y.to(device)
        feature = feature.to(device)
        x_dec = x_dec.to(device)
        y_dec = y_dec.to(device)
        feature_dec = feature_dec.to(device)



        y_in = torch.cat([y[:, :t0] , torch.zeros(y.shape[0], to_predict).cuda()],
                         1)  ## y_in is the one without future value
        #print(y_dec.shape, torch.zeros(y_dec.shape[0], to_predict).shape)
        y_dec = torch.cat([y_dec[:, :dec_len], torch.zeros(y_dec.shape[0], to_predict).cuda()],
                         1)
        if y_in[0, -to_predict] != y_dec[0, -to_predict]:
            print('Decoder Input Error!')
        feature = torch.cat(
            [feature[:, :, :t0], torch.zeros(feature.shape[0], feature.shape[1], to_predict).cuda()], 2)
        feature_dec = torch.cat(
            [feature_dec[:, :, :dec_len], torch.zeros(feature_dec.shape[0], feature_dec.shape[1], to_predict).cuda()], 2)

        attention_masks = attention_masks.to(device)
        dec_mask = dec_mask.to(device)

        #output = model(x, y_in, feature, attention_masks[0])
        output = model(x, y_in, x_dec, y_dec, feature, feature_dec, attention_masks[0], dec_mask[0])

        optimizer.zero_grad()
        #print(output.shape)
        pred = output.squeeze(-1)[:, -to_predict:]

        true = y[:, -to_predict:]

        loss = criterion(pred, true) # use the first to_predict output
        # to avoid future information leakage
        loss.backward()
        optimizer.step()



        train_loss += (loss.detach().item() * x.shape[0])

        n += x.shape[0]

    return train_loss / n


def test_epoch(model, test_dl, t0=96, dec_len = 24, to_predict=24):
    criterion1 = config.criterion
    criterion2 = torch.nn.L1Loss()
    model.eval()
    mse = 0
    mae = 0
    n = 0
    with torch.no_grad():  # eliminate gradient i.e. gradients only exists in train()

        for step, (x, y, feature, x_dec, y_dec, feature_dec, attention_masks, dec_mask) in enumerate(test_dl):
            x = x.to(device)
            y = y.to(device)
            feature = feature.to(device)
            x_dec = x_dec.to(device)
            y_dec = y_dec.to(device)
            feature_dec = feature_dec.to(device)


            y_in = torch.cat([y[:, :t0], torch.zeros(y.shape[0], to_predict).cuda()],
                             1)  ## y_in is the one without future value
            y_dec = torch.cat([y_dec[:, :dec_len], torch.zeros(y_dec.shape[0], to_predict).cuda()],
                              1)
            if y_in[0, -to_predict] != y_dec[0, -to_predict]:
                print('Decoder Input Error!')
            feature = torch.cat(
                [feature[:, :, :t0],
                 torch.zeros(feature.shape[0], feature.shape[1], to_predict).cuda()], 2)
            feature_dec = torch.cat(
                [feature_dec[:, :, :dec_len],
                 torch.zeros(feature_dec.shape[0], feature_dec.shape[1], to_predict).cuda()], 2)

            attention_masks = attention_masks.to(device)
            dec_mask = dec_mask.to(device)

            # output = model(x, y_in, feature, attention_masks[0])

            output = model(x, y_in, x_dec, y_dec, feature, feature_dec, attention_masks[0], dec_mask[0])



            pred = output.squeeze()[:, -to_predict:]
            true = y.squeeze()[:, -to_predict:]

            loss = criterion1(pred, true)
            MAE = criterion2(pred, true)


            mse += (loss.detach().item() * x.shape[0])
            mae += (MAE.detach().item() * x.shape[0])
            n += x.shape[0]

    return mse / n, mae / n, pred, true



def main():

    t0 = config.window
    to_predict = config.to_predict
    dec_len = config.decoder_window
    #train_len = config.train_len

    if config.val:
        train_dataset = GetData(data[0:train_len, 0], data[0:train_len, 1:], t0=t0, to_predict=to_predict,dec_len = dec_len)
        val_dataset = GetData(data[train_len:test_len, 0], data[train_len:test_len, 1:], t0=t0, to_predict=to_predict,dec_len = dec_len)
        test_dataset = GetData(data[test_len:, 0], data[test_len:, 1:], t0=t0, to_predict=to_predict,dec_len = dec_len)
        train_dl = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)
    else:
        train_dataset = GetData(data[0:train_len, 0], data[0:train_len, 1:], t0=t0, to_predict=to_predict,dec_len = dec_len)
        test_dataset = GetData(data[train_len:, 0], data[train_len:, 1:], t0=t0, to_predict=to_predict,dec_len = dec_len)
        train_dl = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    print("Current Dataset: {}\n Current model dimension: {}\n Current predict len: {}".format(data_name,
                                               config.embeddings['out_channels'], config.to_predict))
    # dataloaders


    epochs = config.epoch

    Best_loss_rmse = []
    Best_loss_mae = []

    for exp in range(config.repeats):
        Best_loss_rmse.append(exp + 1)
        Best_loss_mae.append(exp + 1)
        model = TransformerTimeSeries(config, feature_weight=True, embedding_weight=False)
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                                    nesterov=config.nesterov, weight_decay=config.weight_decay)  # Adam optimizer
        if config.scheduler == 'On':
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50, 80, 100, epochs],gamma=0.5)
        train_epoch_loss = []
        val_epoch_mse = []
        trues = []
        preds = []
        test_best_mse = float('inf')
        val_best = float('inf')
        for epoch in tqdm(range(epochs)):

            train_loss = train_epoch(model, optimizer, train_dl, t0, dec_len, to_predict)
            if config.val:

                val_mse, val_mae, _, _ = test_epoch(model, val_dl, t0, dec_len, to_predict)

                if val_best > val_mse:
                    val_best = val_mse
                    test_best_mse, test_best_mae, prediction, ground_truth = test_epoch(model, test_dl, t0, dec_len, to_predict)
                    save_model(config, model, epoch + 1)
                    print('\n--------------------------------------')
                    print('Current best validation MSE: {}'.format(val_mse))
                    print('Test MSE improved to {}'.format(test_best_mse))
                    print('Test RMSE improved to {}'.format(np.sqrt(test_best_mse)))
                    print('Test MAE : {}'.format(test_best_mae))
                    print('Train loss: {} \t'.format(np.mean(train_loss)))
                    print('--------------------------------------')
                trues.append(ground_truth.cpu().numpy())
                preds.append(prediction.cpu().numpy())
                val_epoch_mse.append(val_mse)
            else:
                test_mse, test_mae, prediction, ground_truth = test_epoch(model, test_dl, t0, dec_len,
                                                                                    to_predict)
                if test_mse <= test_best_mse:
                    test_best_mse = test_mse
                    test_best_mae = test_mae
                    save_model(config, model, epoch + 1)
                    print('\n--------------------------------------')
                    print('Test MSE improved to {}'.format(test_best_mse))
                    print('Test RMSE improved to {}'.format(np.sqrt(test_best_mse)))
                    print('Test MAE : {}'.format(test_best_mae))
                    print('Train loss: {} \t'.format(np.mean(train_loss)))
                    print('--------------------------------------')
                trues.append(ground_truth.cpu().numpy())
                preds.append(prediction.cpu().numpy())



            train_epoch_loss.append(np.mean(train_loss))



            np.savetxt("train_loss_list.txt", train_epoch_loss)

            #print("Eval Epoch:{}\nEval_loss:{}\nEval_RMSE:{}".format(epoch + 1, test_loss, np.sqrt(test_loss)))
        true_name = config.root + '\GT' + data_name + '.txt'
        pred_name = config.root + '\pred' + data_name + '.txt'
        np.savetxt(pred_name, np.reshape(preds, (-1,)))
        np.savetxt(true_name, np.reshape(trues, (-1,)))

        Best_loss_rmse.append(np.sqrt(test_best_mse))
        Best_loss_mae.append(test_best_mae)
        best_name = config.data_name+'_'+str(config.embeddings['out_channels'])+'_'+str(config.embeddings['conv_len'])+'_'+str(config.to_predict)+'.txt'
        np.savetxt(best_name, Best_loss_rmse)
        np.savetxt(best_name, Best_loss_mae)

