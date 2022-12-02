import torch
import torch.nn as nn
from timm import create_model


class LargeKernelLayer(nn.Conv2d):
    def forward(self, input: torch.Tensor):
        finput = input.flatten(0, 1)[:, None]
        target = abs(self.weight)
        target = target / target.sum((-1, -2), True)
        joined_kernel = torch.cat([self.weight, target], 0)

        reals = target.new_zeros(
            [1, 1] + [s + p * 2 for p, s in zip(self.padding, input.shape[-2:])]
        )
        reals[
            [slice(None)] * 2
            + [slice(p, -p) if p != 0 else slice(None) for p in self.padding]
        ].fill_(1)
        output, power = torch.nn.functional.conv2d(
            finput, joined_kernel, padding=self.padding
        ).chunk(2, 1)

        ratio = torch.div(*torch.nn.functional.conv2d(reals, joined_kernel).chunk(2, 1))
        power = torch.mul(power, ratio)
        output = torch.mul(output, power)
        return output.unflatten(0, input.shape[:2]).flatten(1, 2)


class LargeKernelModel(nn.Module):
    def __init__(self, config):
        """
        name (str): timm model name, e.g. tf_efficientnet_b2_ns
        """
        super().__init__()

        self.config = config
        # Use timm
        self.model = create_model(
            config.model_base_type,
            pretrained=config.pretrained,
            in_chans=32,
            num_classes=2,
        )

        C, _, H, W = (16, 1, 31, 255)  # state_dict["conv_stem.2.weight"].shape

        self.model.conv_stem = nn.Sequential(
            nn.Identity(),
            nn.AvgPool2d((1, 9), (1, 8), (0, 4), count_include_pad=False),
            LargeKernelLayer(1, C, [H, W], 1, [H // 2, W // 2], 1, 1, False),
            self.model.conv_stem,
        )

    def forward(self, x):
        res = self.model(x)
        res = torch.sigmoid(res)
        res = res.mean(1)
        return res
