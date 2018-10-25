""" Audio encoder """
import math
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    """
    A simple encoder convolutional -> recurrent neural network for
    audio input.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec

    """

    def __init__(self, sample_rate, window_size, out_size):
        super(AudioEncoder, self).__init__()

        self.layer1 = nn.Conv2d(1, 32, kernel_size=(41, 11),
                                padding=(0, 10), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=(21, 11),
                                padding=(0, 0), stride=(2, 1))
        self.batch_norm2 = nn.BatchNorm2d(32)

        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        input_size = int(math.floor(input_size - 41) / 2 + 1)
        input_size = int(math.floor(input_size - 21) / 2 + 1)
        self.input_size = input_size*32
        self.out_proj = nn.Linear(self.input_size, out_size)


    def forward(self, src, lengths=None):
        "See :obj:`onmt.encoders.encoder.EncoderBase.forward()`"
        # (batch_size, 1, nfft, t)
        # layer 1
        src = self.batch_norm1(self.layer1(src[:, :, :, :]))

        # (batch_size, 32, nfft/2, t/2)
        src = F.hardtanh(src, 0, 20, inplace=True)

        # (batch_size, 32, nfft/2/2, t/2)
        # layer 2
        src = self.batch_norm2(self.layer2(src))

        # (batch_size, 32, nfft/2/2, t/2)
        src = F.hardtanh(src, 0, 20, inplace=True)

        batch_size = src.size(0)
        length = src.size(3)
        src = src.view(batch_size, -1, length)
        src = src.transpose(0, 2).transpose(1, 2)
        src = self.out_proj(src)

        return src
