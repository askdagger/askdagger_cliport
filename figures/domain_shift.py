import os
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize
import numpy as np
from matplotlib import pyplot as plt

import askdagger_cliport
from askdagger_cliport.utils import utils


if __name__ == "__main__":
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    utils.set_plot_style()
    results = {}
    interactive_demos = 450
    task = "packing-seen-shapes"
    exp_folder = "exps_domain_shift"
    bias_compensation = True
    fig, ax = plt.subplots(2, 3, figsize=(6.50127, 6.50127 * 0.5))
    for pier in [False, True]:
        fier_list = [False, True] if pier else [True]
        for fier in fier_list:
            relabeling_demos = fier
            validation_demos = fier
            setting = f"rd={relabeling_demos}_pr={pier}_vd={validation_demos}_n={interactive_demos}"
            setting_results = dict(n_annotation=[], success=[])
            for iteration in range(10):
                hydra.core.global_hydra.GlobalHydra.instance().clear()
                os.chdir(os.environ["ASKDAGGER_ROOT"])
                with initialize(config_path="../src/askdagger_cliport/cfg"):
                    vcfg = compose(
                        config_name="eval",
                        overrides=[
                            f"iteration={iteration}",
                            f"interactive_demos={interactive_demos}",
                            f"model_task={task}",
                            f"eval_task={task}",
                            f"exp_folder={exp_folder}",
                            f"relabeling_demos={relabeling_demos}",
                            f"validation_demos={validation_demos}",
                            f"eval.pier={pier}",
                        ],
                    )
                OmegaConf.set_struct(vcfg, False)
                train_results = utils.get_train_results(vcfg)
                if train_results is not None:
                    r = train_results["r"]
                    demos = train_results["demos"]
                    episodes = train_results["episodes"]
                    demo_count = 0
                    max_i = 0
                    max_e = 0
                    begin_i = 0
                    begin_e = 0
                    n_relabeling = 0
                    n_annotation = 0
                    success = []
                    n_annotation = []

                    stop = False
                    shift1 = False
                    shift2 = False
                    while demo_count < interactive_demos:
                        if demo_count >= 150:
                            if not shift1:
                                begin_i = max_i
                                begin_e = max_e
                                shift1 = True
                        if demo_count >= 300:
                            if not shift2:
                                begin_i = max_i
                                begin_e = max_e
                                shift2 = True
                        episode = episodes[max_e]
                        demo = False
                        for step in range(len(episode[3])):
                            if demos[max_i]:
                                demo = True
                            if max_i < len(r) - 1:
                                if r[max_i + 1] == askdagger_cliport.UNKNOWN_RELABELING:
                                    demo = True
                                    n_relabeling += 1
                                    max_i += 1
                            max_i += 1
                        if demo:
                            demo_count += 1
                            r_window = np.asarray(r[begin_i:max_i])
                            demos_window = np.asarray(demos[begin_i:max_i])
                            online_idx = np.asarray(r_window) >= -1
                            annotations = np.sum(r_window[online_idx] == -1)
                            if not validation_demos:
                                annotations += np.sum(np.logical_and(r_window[online_idx] == 1, demos_window[online_idx]))
                            n_annotation.append(annotations)
                            novice_success = np.asarray(
                                [
                                    episode[3][step] > 0
                                    for episode in episodes[begin_e : max_e + 1]
                                    for step in range(len(episode[3]))
                                ]
                            )
                            success.append(np.sum(novice_success[-50:]) / len(novice_success[-50:]))
                        max_e += 1
                        if demo_count < interactive_demos and max_e >= len(episodes):
                            stop = True
                            break
                    if stop:
                        print(f"Not enough demos collected, only {demo_count}/{interactive_demos}")
                        print(f"setting={setting}, task={task}, iteration={iteration}")
                    setting_results["success"].append(success)
                    setting_results["n_annotation"].append(n_annotation)

            if len(setting_results["success"]) == 10:
                label = None
                if pier and fier:
                    label = "ASkDAgger"
                    c_idx = 0
                elif pier:
                    label = "ASkDAgger w/o FIER"
                    c_idx = 2
                elif fier:
                    label = "ASkDAgger w/o PIER"
                    c_idx = 3
                else:
                    label = "Active DAgger"
                    c_idx = 1
                success_mean = np.nanmean(setting_results["success"], axis=0)
                success_std = np.nanstd(setting_results["success"], axis=0)

                # task 1: packing-seen-shapes
                ax[0, 0].set_title("packing-seen-shapes", fontsize=8)
                ax[0, 0].set_ylabel("Novice Success Rate")
                ax[0, 0].plot(success_mean[:150], color=cmap[c_idx])
                ax[0, 0].fill_between(
                    range(len(success_mean[:150])),
                    success_mean[:150] - success_std[:150],
                    success_mean[:150] + success_std[:150],
                    alpha=0.2,
                    color=cmap[c_idx],
                )
                ax[0, 0].set_xlim(0, 150)
                ax[0, 0].set_ylim(0.0, 0.8)

                # task 2: packing-unseen-shapes
                ax[0, 1].set_title("packing-unseen-shapes", fontsize=8)
                ax[0, 1].plot(range(150, 300), success_mean[150:300], color=cmap[c_idx], label=label)
                ax[0, 1].fill_between(
                    range(150, 300),
                    success_mean[150:300] - success_std[150:300],
                    success_mean[150:300] + success_std[150:300],
                    alpha=0.2,
                    color=cmap[c_idx],
                )
                ax[0, 1].set_xlim(150, 300)
                ax[0, 1].set_ylim(0.25, 0.95)

                # task 3: packing-seen-google-objects-seq
                ax[0, 2].set_title("packing-seen-google-objects-seq", fontsize=8)
                ax[0, 2].plot(range(300, 450), success_mean[300:], color=cmap[c_idx])
                ax[0, 2].fill_between(
                    range(300, 450),
                    success_mean[300:] - success_std[300:],
                    success_mean[300:] + success_std[300:],
                    alpha=0.2,
                    color=cmap[c_idx],
                )
                ax[0, 2].set_xlim(300, 450)
                ax[0, 2].set_ylim(0.0, 0.75)

                n_annotation_mean = np.nanmean(setting_results["n_annotation"], axis=0)
                n_annotation_std = np.nanstd(setting_results["n_annotation"], axis=0)

                # task 1: packing-seen-shapes
                ax[1, 0].set_ylabel("Ann. Tuples")
                ax[1, 0].set_xlabel("Demonstrations")
                ax[1, 0].plot(n_annotation_mean[:150], color=cmap[c_idx])
                ax[1, 0].fill_between(
                    range(len(n_annotation_mean[:150])),
                    n_annotation_mean[:150] - n_annotation_std[:150],
                    n_annotation_mean[:150] + n_annotation_std[:150],
                    alpha=0.2,
                    color=cmap[c_idx],
                )
                ax[1, 0].set_xlim(0, 150)
                # ax[1, 0].set_ylim(0, 80)

                # task 2: packing-unseen-shapes
                ax[1, 1].set_xlabel("Demonstrations")
                ax[1, 1].plot(range(150, 300), n_annotation_mean[150:300], color=cmap[c_idx], label=label)
                ax[1, 1].fill_between(
                    range(150, 300),
                    n_annotation_mean[150:300] - n_annotation_std[150:300],
                    n_annotation_mean[150:300] + n_annotation_std[150:300],
                    alpha=0.2,
                    color=cmap[c_idx],
                )
                ax[1, 1].set_xlim(150, 300)
                # ax[1, 1].set_ylim(0, 80)

                # task 3: packing-seen-google-objects-seq
                ax[1, 2].set_xlabel("Demonstrations")
                ax[1, 2].plot(range(300, 450), n_annotation_mean[300:], color=cmap[c_idx])
                ax[1, 2].fill_between(
                    range(300, 450),
                    n_annotation_mean[300:] - n_annotation_std[300:],
                    n_annotation_mean[300:] + n_annotation_std[300:],
                    alpha=0.2,
                    color=cmap[c_idx],
                )
                ax[1, 2].set_xlim(300, 450)
                # ax[1, 2].set_ylim(0, 241)
            else:
                print(f"setting={setting} missing results")

    handles, labels = ax[0, 1].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}", r"\textbf{E}", r"\textbf{F}"]
    for axis, label in zip(ax.flatten(), labels):
        axis.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", transform=axis.transAxes, va="top", ha="right")
    fig.savefig("figures/domain_shift.pdf")
