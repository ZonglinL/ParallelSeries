import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class CausalConv2d(torch.nn.Conv2d):
    '''

    Causal Convolutional Neural Network in 1-D. Basically a special case of CNN-1d
    so its generator inherents from torch.nn.Conv1d class.

    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv2d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           padding=0,
                                           stride=stride,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=True)

        self.__padding = (kernel_size[1] - 1) * dilation
        self.feature_padding = (stride[0]-1) * dilation

    def forward(self, input):
        return super(CausalConv2d, self).forward(F.pad(input, (self.__padding, 0, self.feature_padding, 0)))


class context_embedding(torch.nn.Module):
    '''
       Embedding the context. You can understand this as construct a model using causal CNN.
       However, the return value is sigmoided.
    '''

    def __init__(self,
                 in_channels=1,
                 out_channels=256,  # out_channel
                 kernel=5,
                 stride = (1,1)):
        super(context_embedding, self).__init__()

        self.causal_convolution = CausalConv2d(in_channels,
                                               out_channels,
                                               kernel_size=kernel,
                                               stride = stride)

    def forward(self, x):
        x = self.causal_convolution(x)

        return torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=99999):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Embedding(torch.nn.Module):

    def __init__(self, d_model, conv_len, feature_dim, feature_weight =True, embedding_weight = False):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.conv_len = conv_len
        self.feature_dim = feature_dim
        self.feature_weight = feature_weight
        self.embedding_weight = embedding_weight

        self.input_embedding_y_Q = context_embedding(1, self.d_model, (1, self.conv_len))
        self.input_embedding_y_K = context_embedding(1, self.d_model, (1, self.conv_len))
        self.input_embedding_y_V = context_embedding(1, self.d_model, (1, 1))

        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn2 = torch.nn.BatchNorm2d(self.feature_dim)
        self.input_embedding_y_Q = context_embedding(1, self.d_model, (1, self.conv_len))
        self.input_embedding_y_K = context_embedding(1, self.d_model, (1, self.conv_len))
        self.input_embedding_y_V = context_embedding(1, self.d_model, (1, 1))

        if feature_weight:
            self.input_embedding_f_Q = context_embedding(1, self.d_model, (1, self.conv_len), stride=(1, 1))
            self.input_embedding_f_K = context_embedding(1, self.d_model, (1, self.conv_len), stride=(1, 1))
            self.input_embedding_f_V = context_embedding(1, self.d_model, (1, 1), stride=(1, 1))
            l_dim = int(math.ceil(self.feature_dim / 1))
            self.feature_projection_Q = torch.nn.Linear(l_dim, 1)
            self.feature_projection_K = torch.nn.Linear(l_dim, 1)
            self.feature_projection_V = torch.nn.Linear(l_dim, 1)
        else:
            self.input_embedding_f_Q = context_embedding(1, self.d_model, (self.feature_dim, self.conv_len))
            self.input_embedding_f_K = context_embedding(1, self.d_model, (self.feature_dim, self.conv_len))
            self.input_embedding_f_V = context_embedding(1, self.d_model, (self.feature_dim, 1))



        # positional embedding

        self.positional_embedding = PositionalEmbedding(self.d_model)

    def forward(self, time, target, feature):

        x = time
        B, I = target.shape

        z = target.unsqueeze(1).unsqueeze(1)  # Batch  x input_len, this becomes batch x 1 x1 x input_len
        # reshape it so it can be processed by 2D CNN

        z = self.bn1(z)  # Batch Norm
        input_seq_len = z.shape[-1]  # the length pass to encoder




        _,  f, _ = feature.shape
        if f > 1:
            feature = feature.unsqueeze(1)  # embedding of multi-dim feature, batch x 1 x feature x input_len

        else:
            feature = feature.unsqueeze(1).unsqueeze(1)  # One dim feature

        # input_embedding returns shape
        # (B,f,1,input_len(info_len+pred_len)) -> need (sequence_len,B,f)
        feature = self.bn2(feature.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # below is embedding for target
        # embedding of target, batch x d_model x 1 x input_len

        Qz_embedding = self.input_embedding_y_Q(z).squeeze(2).permute(2, 0, 1)

        # after permute, it becomes input_len x batch x channels(features)
        Kz_embedding = self.input_embedding_y_K(z).squeeze(2).permute(2, 0, 1)
        Vz_embedding = self.input_embedding_y_V(z).squeeze(2).permute(2, 0, 1)

        # below is embedding for feature

        if self.feature_weight:

            Qf_embedding = self.input_embedding_f_Q(feature) # B, C, f, L
            Qf_embedding = Qf_embedding.permute(0, 1, 3, 2)  # multi dim feature, B x C x L x F,
            # after permute, batch x channels x time x features

            # linear combination of features embeddings
            # B, d_model, f, I
            Qf_embedding = self.feature_projection_Q(Qf_embedding).squeeze(-1) # B, C, L

            Qf_embedding = Qf_embedding.permute(2, 0, 1) # L B C


            Kf_embedding = self.input_embedding_f_K(feature)
            Kf_embedding = Kf_embedding.permute(0, 1, 3, 2)
            Kf_embedding = self.feature_projection_K(Kf_embedding).squeeze(-1)
            Kf_embedding = Kf_embedding.permute(2, 0, 1)

            Vf_embedding = self.input_embedding_f_V(feature)
            Vf_embedding = Vf_embedding.permute(0, 1, 3, 2)
            Vf_embedding = self.feature_projection_V(Vf_embedding).squeeze(-1)
            Vf_embedding = Vf_embedding.permute(2, 0, 1)

        else:

            # Qf_embedding = self.input_embedding_f_Q(f).squeeze().permute(2, 0, 1)
            Qf_embedding = self.input_embedding_f_Q(feature).squeeze().permute(2, 0, 1)
            Kf_embedding = self.input_embedding_f_K(feature).squeeze().permute(2, 0, 1)

            Vf_embedding = self.input_embedding_f_V(feature).squeeze().permute(2, 0, 1)

        # get my positional embeddings
        # (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)

        positional_embeddings = self.positional_embedding(x) # 1, L, C

        positional_embeddings = positional_embeddings.permute(1, 0, 2) #L, 1, C
        # from batch x time x feature to time x batch x feature(channels)
        pe_encoder = positional_embeddings[:input_seq_len, :, :]
        # below is to combine embeddings

        if self.embedding_weight:
            W = torch.ones(1, 1).cuda()

            Q_weight = torch.exp(self.embbeding_projection_Q(W))  ## 1x3
            K_weight = torch.exp(self.embbeding_projection_K(W))
            V_weight = torch.exp(self.embbeding_projection_V(W))

            all_embedding_Q = torch.cat([Qz_embedding.unsqueeze(3), pe_encoder.unsqueeze(3), Qf_embedding.unsqueeze(3)],
                                        3)
            ## embedding.shape x 3
            Q = (Q_weight * all_embedding_Q).sum(-1)
            ## 1x3 and embedding.shape x 3 will be reduced to elementwise product along last dimension by pytorch

            all_embedding_K = torch.cat([Kz_embedding.unsqueeze(3), pe_encoder.unsqueeze(3), Kf_embedding.unsqueeze(3)],
                                        3)
            ## embedding.shape x 3
            K = (K_weight * all_embedding_K).sum(-1)

            all_embedding_V = torch.cat([Vz_embedding.unsqueeze(3), pe_encoder.unsqueeze(3), Vf_embedding.unsqueeze(3)],
                                        3)
            ## embedding.shape x 3
            V = (V_weight * all_embedding_V).sum(-1)
        else:
            Q = Qz_embedding + pe_encoder + Qf_embedding  ## add up directly
            K = Kz_embedding + pe_encoder + Kf_embedding
            V = Vz_embedding + pe_encoder + Vf_embedding
        return Q.permute(1, 0 ,2), K.permute(1, 0 ,2), V.permute(1, 0 ,2) # L B C
