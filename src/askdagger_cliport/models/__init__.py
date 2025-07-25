# credit: https://github.com/cliport/cliport

from askdagger_cliport.models.resnet_lat import ResNet45_10s, DropoutResNet45_10s
from askdagger_cliport.models.clip_lingunet_lat import (
    CLIPLingUNetLat,
    DropOutCLIPLingUNetLat,
)


names = {
    # lateral connections
    "plain_resnet_lat": ResNet45_10s,
    "dropout_plain_resnet_lat": DropoutResNet45_10s,
    "clip_lingunet_lat": CLIPLingUNetLat,
    "dropout_clip_lingunet_lat": DropOutCLIPLingUNetLat,
}
