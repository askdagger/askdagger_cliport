# credit: https://github.com/cliport/cliport

import torch
import torch.nn as nn
import torch.nn.functional as F

from askdagger_cliport.models.resnet import IdentityBlock, ConvBlock
from askdagger_cliport.models.core.unet import Up
from askdagger_cliport.models.core.clip import build_model, load_clip, tokenize

from askdagger_cliport.models.core import fusion
from askdagger_cliport.models.core.fusion import FusionConvLat


class CLIPLingUNetLat(nn.Module):
    """CLIP RN50 with U-Net skip connections and lateral connections"""

    def __init__(self, input_shape, output_dim, cfg, device, preprocess, clip_rn50=None):
        super(CLIPLingUNetLat, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg["train"]["batchnorm"]
        self.lang_fusion_type = self.cfg["train"]["lang_fusion_type"]
        self.drop_prob = self.cfg["train"]["drop_prob"]
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess

        if clip_rn50 is None:
            self._load_clip()
        else:
            # self._load_clip()
            self.clip_rn50 = clip_rn50
        self._build_decoder()

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict(), device=self.device).to(self.device)
        for param in self.clip_rn50.parameters():
            param.requires_grad = False
        del model

    def _build_decoder(self):
        # language
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if "word" in self.lang_fusion_type else 1024
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU(inplace=True)
        )
        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion4 = FusionConvLat(input_dim=128 + 64, output_dim=64)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion5 = FusionConvLat(input_dim=64 + 32, output_dim=32)

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion6 = FusionConvLat(input_dim=32 + 16, output_dim=16)

        self.conv2 = nn.Sequential(nn.Conv2d(16, self.output_dim, kernel_size=1))

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, x, lat, l):
        x = self.preprocess(x, dist="clip")

        in_type = x.dtype
        in_shape = x.shape
        x = x[:, :3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        l_enc, l_emb, l_mask = self.encode_text(l)
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3])

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4])

        x = self.layer1(x)
        x = self.lat_fusion4(x, lat[-3])

        x = self.layer2(x)
        x = self.lat_fusion5(x, lat[-2])

        x = self.layer3(x)
        x = self.lat_fusion6(x, lat[-1])

        x = self.conv2(x)

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode="bilinear")
        return x


class DropOutCLIPLingUNetLat(nn.Module):
    """CLIP RN50 with U-Net skip connections and lateral connections and uncertainty estimation."""

    def __init__(self, input_shape, output_dim, cfg, device, preprocess, clip_rn50=None):
        super(DropOutCLIPLingUNetLat, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg["train"]["batchnorm"]
        self.lang_fusion_type = self.cfg["train"]["lang_fusion_type"]
        self.drop_prob = self.cfg["train"]["drop_prob"]
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess

        if clip_rn50 is None:
            self._load_clip()
        else:
            self.clip_rn50 = clip_rn50
        self._build_decoder()

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict(), device=self.device).to(self.device)
        for param in self.clip_rn50.parameters():
            param.requires_grad = False
        del model

    def _build_decoder(self):
        # language
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if "word" in self.lang_fusion_type else 1024
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU(inplace=True)
        )
        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)
        self.lat_fusion1 = FusionConvLat(input_dim=1024 + 512, output_dim=512)
        self.do1 = nn.Dropout2d(p=self.drop_prob)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)
        self.lat_fusion2 = FusionConvLat(input_dim=512 + 256, output_dim=256)
        self.do2 = nn.Dropout2d(p=self.drop_prob)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)
        self.lat_fusion3 = FusionConvLat(input_dim=256 + 128, output_dim=128)
        self.do3 = nn.Dropout2d(p=self.drop_prob)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion4 = FusionConvLat(input_dim=128 + 64, output_dim=64)
        self.do4 = nn.Dropout2d(p=self.drop_prob)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion5 = FusionConvLat(input_dim=64 + 32, output_dim=32)
        self.do5 = nn.Dropout2d(p=self.drop_prob)

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion6 = FusionConvLat(input_dim=32 + 16, output_dim=16)
        self.do6 = nn.Dropout2d(p=self.drop_prob)

        self.conv2 = nn.Sequential(nn.Conv2d(16, self.output_dim, kernel_size=1))

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, x, lat, l):
        x = self.preprocess(x, dist="clip")

        in_type = x.dtype
        in_shape = x.shape
        x = x[:, :3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        l_enc, l_emb, l_mask = self.encode_text(l)
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        assert x.shape[1] == self.input_dim
        x = self.conv1(x)
        x = self.do1(x)

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3])
        x = self.do2(x)

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4])
        x = self.do3(x)

        x = self.layer1(x)
        x = self.lat_fusion4(x, lat[-3])
        x = self.do4(x)

        x = self.layer2(x)
        x = self.lat_fusion5(x, lat[-2])
        x = self.do5(x)

        x = self.layer3(x)
        x = self.lat_fusion6(x, lat[-1])
        x = self.do6(x)

        x = self.conv2(x)
        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode="bilinear")
        return x
