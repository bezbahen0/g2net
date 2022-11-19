import timm
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, name, *, pretrained=False):
        """
        name (str): timm model name, e.g. tf_efficientnet_b2_ns
        """
        super().__init__()

        # Use timm
        model = timm.create_model(name, pretrained=pretrained, in_chans=2)

        clsf = model.default_cfg['classifier']

        
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.cnn = nn.Sequential(
                model,
                nn.Identity(),
               # nn.AdaptiveAvgPool2d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(n_features, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1),
            )
        

    def forward(self, x):
        x = self.cnn(x)
        return x