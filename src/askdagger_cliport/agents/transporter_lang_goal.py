# credit: https://github.com/cliport/cliport

import torch
import numpy as np
from askdagger_cliport.utils import utils
from askdagger_cliport.agents.transporter import TransporterAgent

from askdagger_cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusion
from askdagger_cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion
from askdagger_cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
from askdagger_cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat

from askdagger_cliport.uncertainty_quantification import quantify_uncertainty


class TwoStreamClipLingUNetTransporterAgent(TransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = "plain_resnet_original"
        stream_two_fcn = "clip_lingunet_original"
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp["inp_img"]
        lang_goal = inp["lang_goal"]

        return self.attention.forward(inp_img, lang_goal, softmax=softmax)

    def attn_training_step(self, frame, backprop=True, w=None):
        inp_img = frame["img"]
        lang_goal = frame["lang_goal"]
        attn_label = frame["attn_label"]

        inp = {"inp_img": inp_img, "lang_goal": lang_goal}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, out, attn_label, w=w)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp["inp_img"]
        p0 = inp["p0"]
        lang_goal = inp["lang_goal"]

        return self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)

    def transport_training_step(self, frame, backprop=True, w=None):
        inp_img = frame["img"]
        p0 = frame["p0"]
        lang_goal = frame["lang_goal"]
        transport_label = frame["transport_label"]

        inp = {"inp_img": inp_img, "p0": p0, "lang_goal": lang_goal}
        out = self.trans_forward(inp, softmax=False)
        loss = self.transport_criterion(backprop, out, transport_label, w=w)
        return loss

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        if type(obs) is dict:
            img = self.test_ds.dataset.get_image(obs) if self.test_ds is not None else self.train_ds.dataset.get_image(obs)
        else:
            img = obs
        lang_goal = info["lang_goal"]

        # Attention model forward pass.
        pick_inp = {
            "inp_img": torch.from_numpy(img).to(dtype=torch.float, device=self.device).unsqueeze(0),
            "lang_goal": [lang_goal],
        }
        pick_conf = self.attn_forward(pick_inp)
        assert pick_conf.dim() == 4
        pick_conf = pick_conf.permute(0, 2, 3, 1)
        pick_conf = pick_conf.detach().cpu().numpy()

        pick_uncertainty = quantify_uncertainty(pick_conf.reshape((1, *pick_conf.shape[1:])), **self.uq_kwargs)
        argmax = np.argmax(pick_conf.reshape(pick_conf.shape[0], -1), axis=1)
        coord0, coord1, coord2 = np.unravel_index(argmax, shape=pick_conf.shape[1:])
        p0_pix = np.stack([coord0, coord1], axis=1)
        p0_theta = coord2 * (2 * np.pi / pick_conf.shape[3])
        assert p0_pix.shape[0] == p0_theta.shape[0] == 1
        p0_pix = p0_pix[0]
        p0_theta = p0_theta[0]

        # Transport model forward pass.
        place_inp = {
            "inp_img": torch.from_numpy(img).to(dtype=torch.float, device=self.device).unsqueeze(0),
            "p0": torch.from_numpy(p0_pix).to(dtype=torch.int, device=self.device).unsqueeze(0),
            "lang_goal": [lang_goal],
        }
        place_conf = self.trans_forward(place_inp)
        assert place_conf.dim() == 4
        place_conf = place_conf.permute(0, 2, 3, 1)
        place_conf = place_conf.detach().cpu().numpy()
        place_uncertainty = quantify_uncertainty(place_conf.reshape((1, *place_conf.shape[1:])), **self.uq_kwargs)

        argmax = np.argmax(place_conf.reshape(place_conf.shape[0], -1), axis=1)
        coord0, coord1, coord2 = np.unravel_index(argmax, shape=place_conf.shape[1:])
        p1_pix = np.stack([coord0, coord1], axis=1)
        p1_theta = coord2 * (2 * np.pi / place_conf.shape[3])
        assert p1_pix.shape[0] == p1_theta.shape[0] == 1
        p1_pix = p1_pix[0]
        p1_theta = p1_theta[0]

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        info["pick_uncertainty"] = pick_uncertainty
        info["place_uncertainty"] = place_uncertainty

        return {
            "pose0": (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            "pose1": (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            "pick": [p0_pix[0], p0_pix[1], p0_theta],
            "place": [p1_pix[0], p1_pix[1], p1_theta],
        }


class TwoStreamClipLingUNetLatTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = "plain_resnet_lat"
        stream_two_fcn = "clip_lingunet_lat"
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, self.attention.attn_stream_two),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
