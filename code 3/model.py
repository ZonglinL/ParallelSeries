import torch
from embeddings import *
from Block import *
from Attention import *
import torch.nn.functional as F
import math
import numpy as np


# model


class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper

    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    """

    def __init__(self, config, feature_weight=True, embedding_weight=False):

        super(TransformerTimeSeries, self).__init__()
        self.feature_weight = feature_weight
        self.embedding_weight = embedding_weight
        self.d_model = config.embeddings['out_channels']
        self.num_layers = config.transformer['num_layers']
        self.nhead = config.transformer['num_heads']
        self.conv_len = config.embeddings['conv_len']
        self.dim_feedforward = config.transformer['mlp_dim']
        self.feature_dim = config.feature_dim
        self.to_predict = config.to_predict
        self.t0 = config.window
        self.dropout = config.transformer['dropout']
        self.sparse = config.sparse
        self.dim_feedforward = config.transformer['mlp_dim']
        self.factor = config.factor
        self.n_decoder_layers = config.decoder_layers



        # convolution layers
        self.encoder_embedding = Embedding(self.d_model, self.conv_len, self.feature_dim, feature_weight,
                                           embedding_weight)
        self.decoder_embedding = Embedding(self.d_model, self.conv_len, self.feature_dim, feature_weight,
                                           embedding_weight)

        # dropouts
        self.drop = torch.nn.Dropout(self.dropout)

        # attention
        # sparsity
        if self.sparse == 'prob':
            self.attn = ProbAttention(attention_dropout=self.dropout, factor = self.factor)


        else:
            self.attn = FullAttention(False, factor = self.factor, attention_dropout=self.dropout)


        self.FirstEnc = FirstAttentionLayer(attention=self.attn,
                                            d_model=self.d_model, n_heads=self.nhead)
        self.FirstDec = FirstAttentionLayer(attention=self.attn,
                                            d_model=self.d_model, n_heads=self.nhead)
        self.Att = AttentionLayer(attention=self.attn,
                                  d_model=self.d_model, n_heads=self.nhead)
        self.encoder_layer = Encoder(self.Att, self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, self.num_layers-1)
        if self.n_decoder_layers != 0:
            self.DecoderAtt = AttentionLayer(attention=FullAttention(False, factor = self.factor,
                                         attention_dropout=self.dropout),
                                         d_model=self.d_model, n_heads=self.nhead)




        # linear
        self.fc1 = torch.nn.Linear(self.d_model, self.dim_feedforward)
        self.fc2 = torch.nn.Linear(self.dim_feedforward, self.d_model)
        self.fc4 = torch.nn.Linear(self.d_model, self.dim_feedforward)
        self.fc5 = torch.nn.Linear(self.dim_feedforward, self.d_model)
        self.fc3 = torch.nn.Linear(self.d_model, 1)
        self.norm1 = torch.nn.LayerNorm(self.d_model)
        self.norm2 = torch.nn.LayerNorm(self.d_model)
        self.norm3 = torch.nn.LayerNorm(self.d_model)
        self.norm4 = torch.nn.LayerNorm(self.d_model)
        self.norm5 = torch.nn.LayerNorm(self.d_model)
        self.norm6 = torch.nn.LayerNorm(self.d_model)
        self.norm7 = torch.nn.LayerNorm(self.d_model)

    def forward(self, x_in, y_in, x_out, y_out, feature_in, feature_out, attention_masks, dec_mask):

        seq_len_in = y_in.shape[-1]
        label_len_in = y_out.shape[-1]
        # below is reshaped input
        q_in, k_in, v_in = self.encoder_embedding(x_in, y_in, feature_in)

        q_dec, _, _ = self.decoder_embedding(x_out, y_out, feature_out)

        # below is first layer of transformer
        if self.sparse == 'prob':

            output, _ = self.Att(q_in, k_in, v_in,
                                 attn_mask=attention_masks[:seq_len_in, :seq_len_in])

            output = q_in + self.norm1(self.drop(output))

            output = self.norm2(output + self.drop(self.fc2(F.relu(self.fc1(output)))))

            output, _ = self.encoder(output, attention_masks)


            # Decoder
            # time batch feature
            if self.n_decoder_layers != 0:
                dec_embed_q = q_dec

                output_dec = dec_embed_q + self.norm3(self.drop(self.Att(dec_embed_q, dec_embed_q, dec_embed_q,
                                                                attn_mask=dec_mask[:label_len_in, :label_len_in])[0]
                                                                )
                                                  )
                #B L C
                output = output_dec + self.drop(self.DecoderAtt(output_dec, output, output,
                                                                attn_mask=dec_mask[:label_len_in, :label_len_in])[0])

            x = y = self.norm5(output)
            y = self.norm6(output + self.drop(self.fc5(F.relu(self.fc4(y)))))

            output = self.fc3(self.norm7(x+y))

        ### below is the final linear layer
        else:
            output, _ = self.Att(q_in, k_in, v_in,
                                 attn_mask=attention_masks[:seq_len_in, :seq_len_in])

            output = q_in + self.norm1(self.drop(output))

            output = self.norm2(output + self.drop(self.fc2(F.relu(self.fc1(output)))))

            for i in range(self.num_layers - 1):
                output, _ = self.encoder(output, output, output, attention_masks)


            # Decoder
            # time batch feature
            if self.n_decoder_layers != 0:
                dec_embed_q = q_dec

                output_dec = dec_embed_q + self.norm3(self.drop(self.Att(dec_embed_q, dec_embed_q, dec_embed_q,
                                                                         attn_mask=dec_mask[:label_len_in,
                                                                                   :label_len_in])[0]
                                                                )
                                                      )
                # B L C
                output = output_dec + self.drop(self.DecoderAtt(output_dec, output, output,
                                                                attn_mask=dec_mask[:label_len_in, :label_len_in])[0])

            x = y = self.norm5(output)
            y = self.norm7(output + self.drop(self.fc5(F.relu(self.fc4(y)))))

            output = self.fc3(self.norm6(x + y))

        return output
