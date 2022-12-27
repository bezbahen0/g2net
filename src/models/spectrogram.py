import timm
import torch.nn as nn


class SpectrogramModel(nn.Module):
    def __init__(self, config):
        """
        name (str): timm model name, e.g. tf_efficientnet_b7_ns or inception_v4
        """
        super().__init__()

        self.config = config

        # Use timm
        self.model = timm.create_model(
            config.model_base_type,
            pretrained=config.pretrained,
            in_chans=2,
            drop_rate=config.dropout,
            num_classes=1,
        )
        ## drop_rate=dropout
        #clsf = self.model.default_cfg['classifier']
        #n_features = self.model._modules[clsf].in_features
        #self.model._modules[clsf] = nn.Identity()
        #self.cnn = nn.Sequential(
        #       self.model,
        #       nn.Identity(),
        #      # nn.AdaptiveAvgPool2d((1, 1, 1)),
        #       nn.Flatten(),
        #       nn.Linear(n_features, 512),
        #       nn.ReLU(inplace=True),
        #       nn.Dropout(p=config.dropout),
        #       nn.Linear(512, 1),
        #   )

    def forward(self, x):
        x = self.model(x)
        return x
