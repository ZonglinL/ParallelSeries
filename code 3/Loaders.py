import torch
from torch.utils.data import Dataset




class GetData(Dataset):
    '''
        Get dataset
    '''

    def __init__(self, target, feature, t0=72, to_predict=24,dec_len = 24, transform=None):

        self.t0 = t0  # known time step
        self.to_predict = to_predict
        self.dec_len = dec_len
        self.transform = None
        length = target.shape[0]
        enc_step = t0 + to_predict  ##step in one data point (known + topredict)
        dec_step = dec_len + to_predict
        #print('dec_len', dec_step)

        eff_len = length - enc_step + 1  ## effective length of the whole time series i.e. can be break in to how many data points

        Y = []
        F = []
        Y_dec = []
        F_dec = []
        time = []
        time_dec = []


        for i in range(eff_len):
            Y.append(target[i:enc_step + i].unsqueeze(0))  ## size * time step
            F.append(feature[i:enc_step + i].unsqueeze(0).permute(0, 2, 1))  ## size * feature size * time step
            time.append(torch.arange(i + 1, i + enc_step+1).type(torch.float).unsqueeze(0))

            #print(target[i+t0-dec_len:enc_step + i].shape)
            Y_dec.append(target[i+t0-dec_len:enc_step + i].unsqueeze(0))

            F_dec.append(feature[i + t0 - dec_len:enc_step + i].unsqueeze(0).permute(0, 2, 1))
            time_dec.append(torch.arange(i + 1 +t0-dec_len, i + enc_step+1).type(torch.float).unsqueeze(0))

        self.target = torch.cat(Y)  ## size * dimension
        self.feature = torch.cat(F)
        self.target_dec = torch.cat(Y_dec)  ## size * dimension
        self.feature_dec = torch.cat(F_dec)

        self.time = torch.cat(time)  ## size * dimension

        self.time_dec = torch.cat(time_dec)

        self.mask_enc = self._generate_square_subsequent_mask(t0, to_predict)
        self.mask_dec = self._generate_square_subsequent_mask(dec_len, to_predict)

        # print out shapes to confirm desired output
        print(self.time.shape)
        print(self.target.shape)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.time[idx, :],
                  self.target[idx, :],
                  self.feature[idx, :],
                  self.time_dec[idx, :],
                  self.target_dec[idx, :],
                  self.feature_dec[idx, :],
                  self.mask_enc,
                  self.mask_dec)  # in tuple shape, will be called in training, eval and testing.

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _generate_square_subsequent_mask(self, t0, to_predict):
        mask = torch.zeros(t0 + to_predict, t0 + to_predict)
        for i in range(0, t0 + to_predict):
            mask[i, i + 1:] = 1
            ## for known days, mask does not matter
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # .masked_fill(mask == 1, float(0.0))
        ##maks current and future for furture predictions
        return mask
