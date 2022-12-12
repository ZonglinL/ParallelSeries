import torch
import torch.nn.functional as F
from embeddings import *
import numpy as np
from math import sqrt


class conv_attn(torch.nn.Module):

    def __init__(self, d_model: int, nhead: int, conv_len, dim_feedforward: int = 2048, dropout: float = 0.01,
                 activation=F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(conv_attn, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                     **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.bn1 = torch.nn.BatchNorm2d(d_model)
        self.bn2 = torch.nn.BatchNorm2d(d_model)
        self.bn3 = torch.nn.BatchNorm2d(d_model)

        # Legacy string support for activation function.
        self.activation = activation

        self.input_embedding_Q = context_embedding(d_model, d_model, (1, conv_len))
        self.input_embedding_K = context_embedding(d_model, d_model, (1, conv_len))
        self.input_embedding_V = context_embedding(d_model, d_model, (1, 1))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(conv_attn, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        ## time batch channel

        x = x.permute(1, 2, 0).unsqueeze(2)  # batch x channel x 1 x time

        # Q = self.bn1(self.input_embedding_Q(x)).squeeze().permute(2,0,1)
        # K = self.bn2(self.input_embedding_K(x)).squeeze().permute(2,0,1)
        # V = self.bn3(self.input_embedding_V(x)).squeeze().permute(2,0,1)
        Q = self.input_embedding_Q(x).squeeze().permute(2, 0, 1)
        K = self.input_embedding_K(x).squeeze().permute(2, 0, 1)
        V = self.input_embedding_V(x).squeeze().permute(2, 0, 1)

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(Q, K, V, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                  attn_mask: [torch.Tensor], key_padding_mask: [torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(Q, K, V,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

    # self-attention block
    def _sa_block(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                  attn_mask: [torch.Tensor], key_padding_mask: [torch.Tensor]) -> torch.Tensor:
        x, attn = self.self_attn(Q, K, V,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout(x), attn

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


#  ProbSpar from informer
class ProbMask:
    def __init__(self, B, H, L, index, scores, device=torch.device('cuda:0')):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)

        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])

        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask



class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            # print(scores[0].shape)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape  # Batch Length 1 dmodel
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        # print(context.shape) # Batch x 1 x Length x dodel
        return context.transpose(2, 1).contiguous(), attn


class Encoder(nn.Module):
    def __init__(self, attention, d_model, n_heads, dim_feedforward=2048, dropout=0,
                 d_keys=None, d_values=None, mix=False):
        super(Encoder, self).__init__()
        self.dropout = dropout

        self.inner_attention = attention

        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.mix = mix

        self.fc1 = torch.nn.Linear(d_model, dim_feedforward)
        self.fc2 = torch.nn.Linear(dim_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        ##dropouts
        self.drop1 = torch.nn.Dropout(self.dropout)
        self.drop2 = torch.nn.Dropout(self.dropout)
        self.drop3 = torch.nn.Dropout(self.dropout)

    def forward(self, q, k, v, attn_mask):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads


        out, attn = self.inner_attention(
            q,
            k,
            v,
            attn_mask
        )
        out = out + q
        out = self.norm1(self.drop1(out))

        out = self.norm2(out + self.drop3(self.fc2(self.drop2(F.relu(self.fc1(out))))))

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FirstAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(FirstAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention


        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)


        return self.out_projection(out), attn
