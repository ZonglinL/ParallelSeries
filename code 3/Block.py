
import torch
import torch.nn as nn
import math
import copy






class TransformerEncoder(torch.nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None)\
            -> torch.Tensor:

        output = src

        for mod in self.layers:
            output,_ = mod(output, output, output, mask)

        return output,_


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class InformerDecoder(torch.nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super(InformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None)\
            -> torch.Tensor:

        output = src

        for mod in self.layers:
            output,_ = mod(output, mask)

        return output,_