import os
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize
import numpy as np
from matplotlib import pyplot as plt

import askdagger_cliport
from askdagger_cliport.utils import utils


if __name__ == "__main__":
    utils.set_plot_style()
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    results = {}
    interactive_demos = 150
    tasks = [
        "insert_bolts",
    ]
    bias_compensation = True
    exp_folder = "exps_real"
    fig, ax = plt.subplots(1, 3, figsize=(6.50127, 0.3 * 6.50127))
    ax_idx = -1
    setting_results = dict(annotated=[], validated=[], relabeled=[], novice_success=[], system_success=[], sensitivity=[])
    for task in tasks:
        ax_idx += 1
        relabeling_demos = True
        validation_demos = True
        pier = True
        setting = f"rd={relabeling_demos}_pr={pier}_vd={validation_demos}_n={interactive_demos}"
        for iteration in range(1):
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
                n_relabeling = 0
                n_annotation = 0
                n_validation = 0
                relabeled = []
                annotated = []
                validated = []
                n_success = []
                s_success = []
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
                        if demo_count % 10 == 0:
                            annotated.append(n_annotation)
                            validated.append(n_validation)
                            relabeled.append(n_relabeling)
                        novice_success = np.asarray(
                            [episode[3][step] > 0 for episode in episodes[: max_e + 1] for step in range(len(episode[3]))]
                        )
                        system_success = np.asarray(
                            [episode[1][step] > 0 for episode in episodes[: max_e + 1] for step in range(len(episode[1]))]
                        )
                        novice_failure = np.logical_not(novice_success)
                        r_window = np.asarray(r[:max_i])
                        online_idx = np.asarray(r_window) >= -1
                        queries = np.logical_or(r_window[online_idx] == -1, r_window[online_idx] == 1)
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
                        n_success.append(np.sum(novice_success[-50:]) / len(novice_success[-50:]))
                        s_success.append(np.sum(system_success[-50:]) / len(system_success[-50:]))
                        sens.append(sensitivity)
                    max_e += 1
                    if demo_count < interactive_demos and max_e >= len(episodes):
                        stop = True
                        break
                if stop:
                    print(f"Not enough demos collected, only {demo_count}/{interactive_demos}")
                    print(f"setting={setting}, task={task}, iteration={iteration}")
                setting_results["annotated"].append(annotated)
                setting_results["validated"].append(validated)
                setting_results["relabeled"].append(relabeled)
                setting_results["novice_success"].append(n_success)
                setting_results["system_success"].append(s_success)
                setting_results["sensitivity"].append(sens)

        if len(setting_results["annotated"]) == 1:
            s = "-"
            offset = 0
            o_color = "r"
            ax[0].plot(range(1, 151), setting_results["novice_success"][0], color=cmap[0], label="Novice")
            ax[0].plot(range(1, 151), setting_results["system_success"][0], color=cmap[5], label="System")
            ax[0].legend()
            ax[0].set_xlabel("Demonstrations")
            ax[0].set_ylabel("Success Rate")
            ax[0].set_ylim(0, 1.05)

            mean_annotation = np.mean(setting_results["annotated"], axis=0)
            ax[1].bar(
                10 * np.arange(1, 1 + len(mean_annotation)) + offset,
                mean_annotation,
                color=cmap[0],
                width=10,
                label="Ann.",
                edgecolor=o_color
            )
            mean_relabeled = np.mean(setting_results["relabeled"], axis=0)

            ax[1].bar(
                10 * np.arange(1, 1 + len(mean_annotation)) + offset,
                mean_relabeled,
                color="k",
                bottom=mean_annotation,
                width=10,
                label="Rel.",
                alpha=0.5,
                edgecolor="k",
            )
            mean_validated = np.mean(setting_results["validated"], axis=0)
            ax[1].bar(
                10 * np.arange(1, 1 + len(mean_annotation)) + offset,
                mean_validated,
                color="g",
                bottom=mean_annotation + mean_relabeled,
                width=10,
                label="Val.",
                alpha=0.5,
                edgecolor="g",
            )
            ax[1].set_xlabel("Demonstrations")
            ax[1].set_ylabel("Demonstration Tuples")
            ax[1].legend(loc="upper left")

            ax[2].plot(range(1, 151), setting_results["sensitivity"][0], color=cmap[0], label="Sensitivity")
            ax[2].plot(range(1, 151), [0.9] * 150, color="k", linestyle="--", label="Desired")
            ax[2].set_xlabel("Demonstrations")
            ax[2].set_ylabel("Sensitivity")
            ax[2].set_ylim(0.6, 1.05)
            ax[2].legend()

    fig.tight_layout(rect=[0, 0, 1, 1])
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}"]
    for axis, label in zip(ax.flatten(), labels):
        axis.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", transform=axis.transAxes, va="top", ha="right")
    plt.savefig("figures/real.pdf")
