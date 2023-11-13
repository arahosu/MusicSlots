import torch
from torch import nn
from torch.nn import functional as F
import math

class SimpleDecoder(nn.Module):
    """
    Baseline CNN Decoder described in Locatello et al. (2020):
    https://arxiv.org/pdf/2006.15055.pdf
    """
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 decoder_initial_size: list,
                 kernel_size: int = 5,
                 num_strided_decoder_layers: int = 4,
                 return_alpha_mask: bool = True,
                 use_deconv=True):
        super().__init__()

        blocks = []

        for i in range(num_strided_decoder_layers):
            if use_deconv:
                conv_layer = nn.ConvTranspose2d(
                    c_out, out_channels=c_out, kernel_size=kernel_size,
                    stride=2, padding=2, output_padding=1)
            else:
                conv_layer = nn.Sequential(
                    nn.Upsample(scale_factor=(2, 2), mode='bilinear'),
                    nn.Conv2d(c_out, out_channels=c_out, kernel_size=kernel_size, stride=1, padding='same')
                )
            blocks.append(
                conv_layer
            )
            
            blocks.append(nn.ReLU())

        if use_deconv:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_out, out_channels=c_out, kernel_size=kernel_size,
                        stride=1, padding=2),
                    nn.ReLU()
                )
            )

            blocks.append(
                nn.ConvTranspose2d(
                    c_out, out_channels=c_in + 1 if return_alpha_mask else c_in,
                    kernel_size=3, stride=1, padding=1)
            )
        else:
            out_conv = nn.Sequential(
                nn.Conv2d(
                    c_out, out_channels=c_out, kernel_size=kernel_size,
                    stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(
                    c_out, out_channels=c_in + 1 if return_alpha_mask else c_in,
                    kernel_size=3, stride=1, padding='same')
            )
            blocks.append(out_conv)
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)