import numpy as np
from scipy.stats import entropy
from typing import Union


def quantify_uncertainty(P: np.ndarray, measure: str = "entropy", norm_ord: Union[float, str] = 2, **kwargs) -> float:
    """
    Quantify uncertainty of a prediction or set of predictions.

    Parameters
    ----------
    P : np.ndarray
        Prediction or set of predictions.
    measure : str, optional
        Uncertainty measure to use. Options are "confidence", "margin", "entropy", "vote_entropy", "consensus_entropy", "kl", "jeffreys", "prediction_variance", "conservative_confidence", "novelty", by default "entropy".
    norm_ord : Union[float, str], optional
        Order of norm to use for combining uncertainty measures for multiple predictions, by default "inf".
    **kwargs
        Additional keyword arguments for uncertainty measures.

    Returns
    -------
    float
        Uncertainty measure.
    """
    if len(P) == 1:
        assert measure in [
            "confidence",
            "margin",
            "entropy",
        ], "Only confidence, margin and entropy measures are supported for single predictions."
        P = P[0]
    else:
        assert measure in [
            "vote_entropy",
            "consensus_entropy",
            "kl",
            "jeffreys",
            "prediction_variance",
            "conservative_confidence",
            "novelty",
        ], "Only vote_entropy, consensus_entropy, kl, jeffreys, prediction_variance, conservative_confidence, geomloss and novelty measures are supported for multiple predictions."
        Pc = np.mean(P, axis=0)
    if measure == "confidence":
        return 1 - np.max(P)
    elif measure == "margin":
        sorted_P = np.sort(P.flatten())
        return 1 - (sorted_P[-1] - sorted_P[-2])
    elif measure == "entropy":
        return entropy(P.flatten())
    elif measure == "vote_entropy":
        C = len(P)
        V = np.zeros(P[0].shape)
        for vm in P:
            V[np.argmax(vm)] += 1
        return entropy(V / C)
    elif measure == "consensus_entropy":
        return entropy(Pc.flatten())
    elif measure == "kl":
        measures = [entropy(vm.flatten(), Pc.flatten()) for vm in P]
    elif measure == "novelty":
        entr = entropy(P, axis=0)
        return np.nanmean(entr)
    elif measure == "jeffreys":
        measures = [entropy(vm.flatten(), Pc.flatten()) + entropy(Pc.flatten(), vm.flatten()) for vm in P]
    elif measure == "prediction_variance":
        argmax = np.argmax(Pc)
        return np.var(P[:, argmax])
    elif measure == "conservative_confidence":
        argmax = np.argmax(Pc)
        return 1 - (Pc.flatten()[argmax] - np.var(P[:, argmax]))
    else:
        raise ValueError("Measure not implemented.")
    if len(measures) == 1:
        return measures[0]
    else:
        assert norm_ord is not None, "Norm order must be specified for your settings."
        return np.linalg.norm(measures, ord=norm_ord)
