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
    algos = {
        "ASkDAgger": cmap[0],
        "ActiveDAgger": cmap[1],
        "ThriftyDAgger": cmap[6],
        "SafeDAgger": cmap[7],
    }
    interactive_demos = 300
    tasks = [
        "packing-seen-google-objects-seq",
        "packing-seen-google-objects-group",
        "packing-seen-shapes",
        "put-block-in-bowl-seen-colors",
    ]
    fig, ax = plt.subplots(1, 4, figsize=(6.50127, 0.35 * 6.50127))
    offset = -50
    for askdagger in [False, True]:
        relabeling_demos = askdagger
        validation_demos = askdagger
        pier = askdagger
        exp_folders = ["exps"] if askdagger else ["exps", "exps_thrifty", "exps_safe"]
        for exp_folder in exp_folders:
            setting = f"rd={relabeling_demos}_pr={pier}_vd={validation_demos}_n={interactive_demos}_f={exp_folder}"

            offset += 20
            ax_idx = -1
            overrides_dict={
                "interactive_demos": interactive_demos,
                "relabeling_demos": relabeling_demos,
                "validation_demos": validation_demos,
                "eval.pier": pier,
                "exp_folder": exp_folder,
            }
            if exp_folder == "exps_thrifty":
                overrides_dict["agent"] = "dropout_cliport"
            for task in tasks:
                setting_results = dict(annotated=[], validated=[], relabeled=[])
                ax_idx += 1
                
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

                if ax_idx == 0:
                    if askdagger:
                        algo = "ASkDAgger"
                        hatch = None
                    elif exp_folder == "exps":
                        algo = "ActiveDAgger"
                        hatch = "\\\\"
                    elif exp_folder == "exps_thrifty":
                        algo = "ThriftyDAgger"
                        hatch = "--"
                    elif exp_folder == "exps_safe":
                        algo = "SafeDAgger"
                        hatch = "//"
                    else:
                        raise ValueError(f"Unknown exp_folder: {exp_folders}")
                    color = algos[algo]
                
                ax[ax_idx].set_xlabel("Demonstrations")
                if ax_idx == 0:
                    ax[ax_idx].set_ylabel("Demonstration Tuples")
                
                for iteration in range(10):
                    hydra.core.global_hydra.GlobalHydra.instance().clear()
                    os.chdir(os.environ["ASKDAGGER_ROOT"])
                    overrides_dict["iteration"] = iteration
                    overrides_dict["model_task"] = task
                    overrides_dict["eval_task"] = task
                    overrides = [f"{key}={value}" for key, value in overrides_dict.items()]
                    with initialize(config_path="../src/askdagger_cliport/cfg"):
                        vcfg = compose(
                            config_name="eval",
                            overrides=overrides,
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
                                if demo_count % 100 == 0:
                                    annotated.append(n_annotation)
                                    validated.append(n_validation)
                                    relabeled.append(n_relabeling)
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

                hatch = None
                if len(setting_results["annotated"]) == 10:
                    # Annotation demos
                    o_color = "r"
                    mean_annotated = np.mean(setting_results["annotated"], axis=0)
                    label = f"{algo}: Annotation"
                    ax[ax_idx].bar(
                        100 * np.arange(1, 1 + len(mean_annotated)) + offset,
                        mean_annotated,
                        color=color,
                        width=20,
                        hatch=hatch,
                        label=label,
                        edgecolor="r",
                        lw=0.5,
                    )

                    # Relabeled demos
                    if relabeling_demos:
                        mean_relabeled = np.mean(setting_results["relabeled"], axis=0)
                        label = "ASkDAgger: Relabeled"
                        ax[ax_idx].bar(
                            100 * np.arange(1, 1 + len(mean_annotated)) + offset,
                            mean_relabeled,
                            color="k",
                            bottom=mean_annotated,
                            width=20,
                            hatch=hatch,
                            label=label,
                            edgecolor="k",
                            lw=0.5,
                            alpha=0.3,
                        )

                    # Validation demos
                    if validation_demos:
                        mean_validated = np.mean(setting_results["validated"], axis=0)
                        label = "ASkDAgger: Validation"
                        ax[ax_idx].bar(
                            100 * np.arange(1, 1 + len(mean_annotated)) + offset,
                            mean_validated,
                            color="g",
                            bottom=mean_annotated + mean_relabeled,
                            width=20,
                            hatch=hatch,
                            label=label,
                            edgecolor="g",
                            lw=0.5,
                            alpha=0.3,
                        )
                else:
                    print(f"setting={setting}, task={task} missing results")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}"]
    for axis, label in zip(ax.flatten(), labels):
        axis.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", transform=axis.transAxes, va="top", ha="right")
    plt.savefig("figures/demo_types.pdf")
