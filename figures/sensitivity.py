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
    interactive_demos = 300
    tasks = [
        "packing-seen-google-objects-seq",
        "packing-seen-google-objects-group",
        "packing-seen-shapes",
        "put-block-in-bowl-seen-colors",
    ]
    fig, ax = plt.subplots(1, 4, figsize=(6.50127, 0.2 * 6.50127))
    ax_idx = -1
    for task in tasks:
        setting_results = dict(sens=[])
        ax_idx += 1
        relabeling_demos = True
        validation_demos = True
        pier = True
        setting = f"rd={relabeling_demos}_pr={pier}_vd={validation_demos}_n={interactive_demos}"
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
                n_relabeling = 0
                n_annotation = 0
                n_validation = 0
                sens = []
                stop = False
                while demo_count < interactive_demos:
                    episode = episodes[max_e]
                    demo = False
                    for step in range(len(episode[3])):
                        if demos[max_i]:
                            demo = True
                            if validation_demos and r[max_i] == askdagger_cliport.KNOWN_SUCCESS:
                                n_validation += 1
                            elif not validation_demos or r[max_i] == askdagger_cliport.KNOWN_FAILURE:
                                n_annotation += 1
                        if max_i < len(r) - 1:
                            if r[max_i + 1] == askdagger_cliport.UNKNOWN_RELABELING:
                                demo = True
                                n_relabeling += 1
                                max_i += 1
                        max_i += 1
                    if demo:
                        demo_count += 1
                        r_window = np.asarray(r[:max_i])
                        online_idx = np.asarray(r_window) >= -1
                        demos_window = np.asarray(demos[:max_i])
                        queries = np.logical_or(r_window[online_idx] == -1, r_window[online_idx] == 1)
                        novice_success = np.asarray(
                            [episode[3][step] > 0 for episode in episodes[: max_e + 1] for step in range(len(episode[3]))]
                        )
                        novice_failure = np.logical_not(novice_success)
                        tn = np.logical_and(novice_success, np.logical_not(queries))
                        fp = np.logical_and(novice_success, queries)
                        tp = np.logical_and(np.logical_not(novice_success), queries)
                        fn = np.logical_and(np.logical_not(novice_success), np.logical_not(queries))
                        failure_window = np.where(novice_failure)[0][-50:]
                        sensitivity = (
                            np.sum(tp[failure_window]) / np.sum(novice_failure[failure_window])
                            if np.sum(novice_failure[failure_window]) > 0
                            else np.NaN
                        )
                        sens.append(sensitivity)
                    max_e += 1
                    if demo_count < interactive_demos and max_e >= len(episodes):
                        stop = True
                        break
                if stop:
                    print(f"Not enough demos collected, only {demo_count}/{interactive_demos}")
                    print(f"setting={setting}, task={task}, iteration={iteration}")
                setting_results["sens"].append(sens)

        if len(setting_results["sens"]) == 10:
            label = "AID" if relabeling_demos and pier and validation_demos else "BL"
            color = cmap[0] if relabeling_demos else cmap[1]

            mean_sens = np.mean(setting_results["sens"], axis=0)
            std_sens = np.std(setting_results["sens"], axis=0)

            ax[ax_idx].set_xlabel("Demonstrations")
            if ax_idx == 0:
                ax[ax_idx].set_ylabel("Sensitivity")

            title_split = task.split("-")
            if "seen" in title_split:
                title_split.remove("seen")
            if "colors" in title_split:
                title_split.remove("colors")
            title = title_split[0]
            for i, split in enumerate(title_split[1:]):
                if i % 2 == 0:
                    title += f"-{split}"
                else:
                    title += f"\n{split}"
            ax[ax_idx].set_title(title)
            ax[ax_idx].plot(
                range(len(mean_sens)),
                mean_sens,
                color=color,
                label="Sensitivity",
            )
            ax[ax_idx].fill_between(
                range(len(mean_sens)),
                mean_sens - std_sens,
                mean_sens + std_sens,
                alpha=0.2,
                color=color,
            )
            ax[ax_idx].set_ylim([0.7, 1.05])
            ax[ax_idx].plot([0, len(mean_sens)], [0.9, 0.9], "k--", label="Desired")
        else:
            print(f"setting={setting}, task={task} missing results")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}"]
    for axis, label in zip(ax.flatten(), labels):
        axis.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", transform=axis.transAxes, va="top", ha="right")
    plt.savefig("figures/sensitivity.pdf")
