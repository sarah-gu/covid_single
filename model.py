import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiOutputModel(nn.Module):
    def __init__(self, n_month_classes):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier
        
        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.month = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_month_classes)
        )
    #    self.season = nn.Sequential(
    #        nn.Dropout(p=0.2),
    #        nn.Linear(in_features=last_channel, out_features=n_season_classes)
    #    )
    #    self.hour = nn.Sequential(
    #        nn.Dropout(p=0.2),
    #        nn.Linear(in_features=last_channel, out_features=n_hour_classes)
    #    )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

#        return {
#            'month': self.month(x)
     #       'season': self.season(x),
     #       'hour': self.hour(x)
#        }
        return self.month(x)

    def get_loss(self, net_output, ground_truth):
        month_loss = F.cross_entropy(net_output, ground_truth['month_labels'])
       # season_loss = F.cross_entropy(net_output['season'], ground_truth['season_labels'])
       # hour_loss = F.cross_entropy(net_output['hour'], ground_truth['hour_labels'])
        loss = month_loss
        return loss, {'month': month_loss}
