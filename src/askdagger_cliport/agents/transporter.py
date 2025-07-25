# credit: https://github.com/cliport/cliport

import os
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from askdagger_cliport.tasks import cameras
from askdagger_cliport.utils import utils
from askdagger_cliport.models.core.attention import Attention
from askdagger_cliport.models.streams.two_stream_attention import TwoStreamAttention
from askdagger_cliport.models.streams.two_stream_transport import TwoStreamTransport

from askdagger_cliport.models.streams.two_stream_attention import TwoStreamAttentionLat
from askdagger_cliport.models.streams.two_stream_transport import TwoStreamTransportLat

from askdagger_cliport.uncertainty_quantification import quantify_uncertainty


class TransporterAgent(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)
        self.u = None
        self.device_type = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for mode in ["train", "train_interactive", "eval"]:
            if mode in cfg:
                if "gpu" in cfg[mode] and cfg[mode]["gpu"] != 0:
                    self.device_type = torch.device("cuda")
                else:
                    self.device_type = torch.device("cpu")
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg["train"]["task"]
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = cfg["train"]["n_rotations"]
        if "train_interactive" in cfg:
            self.grad_clip = cfg["train_interactive"]["grad_clip"]
            self.lr = cfg["train_interactive"]["lr"]
            self.weight_decay = cfg["train_interactive"]["weight_decay"]
        elif "train" in cfg:
            self.lr = cfg["train"]["lr"]
            self.grad_clip = cfg["train"]["grad_clip"]
            self.weight_decay = cfg["train"]["weight_decay"]
        self.uq_kwargs = {}
        for mode in ["eval", "train_interactive", "train"]:
            if mode in cfg:
                if "measure" in cfg[mode]:
                    self.uq_kwargs["measure"] = cfg[mode]["measure"]
                    break
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg["train"]["val_repeats"]
        self.save_steps = cfg["train"]["save_steps"]

        self._build_model()
        self._optimizers = {
            "attn": torch.optim.Adam(self.attention.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            "trans": torch.optim.Adam(self.transport.parameters(), lr=self.lr, weight_decay=self.weight_decay),
        }

    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction="mean", w=None):
        loss = F.cross_entropy(pred, labels, reduction="none")
        if w is not None:
            w = w.reshape(*loss.shape)
            loss = loss * w
        if reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp["inp_img"]

        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_training_step(self, frame, backprop=True, w=None):
        inp_img = frame["img"]
        attn_label = frame["attn_label"]

        inp = {"inp_img": inp_img}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, out, attn_label, w=w)

    def attn_criterion(self, backprop, out, label, w=None):
        # Get loss.
        loss = self.cross_entropy_with_logits(out, label, w=w)

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers["attn"]
            self.manual_backward(loss, attn_optim)
            total_norm = 0.0
            for p in self.attention.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.log("tr/attention_grad_norm", total_norm, on_step=True, on_epoch=False)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.grad_clip)
            attn_optim.step()
            total_norm = 0.0
            for p in self.attention.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.log("tr/attention_grad_norm_clipped", total_norm, on_step=True, on_epoch=False)
            attn_optim.zero_grad()
        return loss

    def trans_forward(self, inp, softmax=True):
        inp_img = inp["inp_img"]
        p0 = inp["p0"]
        output = self.transport.forward(inp_img, p0, softmax=softmax)
        return output

    def transport_training_step(self, frame, backprop=True, w=None):
        inp_img = frame["img"]
        p0 = frame["p0"]
        transport_label = frame["transport_label"]

        inp = {"inp_img": inp_img, "p0": p0}
        output = self.trans_forward(inp, softmax=False)
        loss = self.transport_criterion(backprop, output, transport_label, w=w)
        return loss

    def transport_criterion(self, backprop, output, label, w=None):
        # # Get loss.
        output = output.reshape(output.shape[0], np.prod(output.shape[1:]))
        loss = self.cross_entropy_with_logits(output, label, w=w)
        if backprop:
            transport_optim = self._optimizers["trans"]
            self.manual_backward(loss, transport_optim)
            total_norm = 0.0
            for p in self.transport.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.log("tr/transport_grad_norm", total_norm, on_step=True, on_epoch=False)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.transport.parameters(), self.grad_clip)
            total_norm = 0.0
            for p in self.transport.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            self.log("tr/transport_grad_norm_clipped", total_norm, on_step=True, on_epoch=False)
            transport_optim.step()
            transport_optim.zero_grad()

        self.transport.iters += 1
        return loss

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()

        frame = batch
        w = frame["w"] if "w" in frame else None

        # Get training losses.
        step = self.total_steps + 1
        loss0 = self.attn_training_step(frame, w=w)

        if isinstance(self.transport, Attention):
            loss1 = self.attn_training_step(frame, w=w)
        else:
            loss1 = self.transport_training_step(frame, w=w)
        total_loss = loss0 + loss1

        self.log("tr/attn/loss", loss0, on_step=True, on_epoch=False)
        self.log("tr/trans/loss", loss1, on_step=True, on_epoch=False)
        self.log("tr/loss", total_loss, on_step=True, on_epoch=False)
        self.total_steps = step
        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.total_steps
        if global_step in self.save_steps:
            steps = f"{global_step:05d}"
            filename = f"steps={steps}.ckpt"
            checkpoint_path = os.path.join(self.cfg["train"]["train_dir"], "checkpoints")
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if global_step % 1000 == 0:
            # save lastest checkpoint
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg["train"]["train_dir"], "checkpoints")
        ckpt_path = os.path.join(checkpoint_path, "last.ckpt")
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame = batch
            l0 = self.attn_training_step(frame, backprop=False)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1 = self.attn_training_step(frame, backprop=False)
                loss1 += l1
            else:
                l1 = self.transport_training_step(frame, backprop=False)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        if not "train_interactive" in self.cfg:
            utils.set_seed(self.trainer.current_epoch + 1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v["val_loss"].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v["val_loss0"].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v["val_loss1"].item() for v in all_outputs])

        self.log("vl/attn/loss", mean_val_loss0)
        self.log("vl/trans/loss", mean_val_loss1)
        self.log("vl/loss", mean_val_total_loss)

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.dataset.get_image(obs) if self.test_ds is not None else self.train_ds.dataset.get_image(obs)

        # Attention model forward pass.
        pick_inp = {
            "inp_img": torch.from_numpy(img).to(dtype=torch.float, device=self.device).unsqueeze(0),
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
            "pick": p0_pix,
            "place": p1_pix,
        }

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu, using_native_amp, using_lbfgs
    ):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds if self.test_ds is not None else self.train_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)["state_dict"])
        self.to(device=self.device_type)


class TwoStreamClipUNetTransporterAgent(TransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = "plain_resnet"
        stream_two_fcn = "clip_unet"
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipUNetLatTransporterAgent(TransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = "plain_resnet_lat"
        stream_two_fcn = "clip_unet_lat"
        self.attention = TwoStreamAttentionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
