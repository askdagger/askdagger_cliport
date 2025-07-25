import numpy as np
from sklearn.metrics import roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression


def sag(
    u,
    r,
    k,
    s_des=0.9,
    p_rand=0.0,
    u_normalization=True,
    impute=True,
    n_min=1,
    n_rep=10,
    mode="sensitivity",
):
    """
    Get the threshold for the uncertainty measure that achieves a desired sensitivity, specificity, or system success rate.
    :param u: list of uncertainties
    :param r: list of teacher rewards
    :param k: list of model update counts
    :param s_des: desired sensitivity
    :param p_rand: rate of random queries
    :param u_normalization: normalize for shifting uncertainty because of model updates
    :param impute: impute missing labels
    :param n_min: minimum number of negative samples
    :param n_rep: number of repetitions for imputation
    :param mode: gating mode, must be in ["sensitivity", "specificity", "success"]
    :return: gating threshold
    """
    assert mode in [
        "sensitivity",
        "specificity",
        "success",
    ], f"Invalid mode: {mode}, should be sensitivity, specificity or success."
    window = 0
    u_array = np.array(u, dtype=float)
    r_array = np.asarray(r, dtype=int)
    k_array = np.asarray(k, dtype=int)

    window_idx = np.logical_and(r_array > -2, k_array >= k_array[-1] - window)
    u_window = u_array[window_idx]
    r_window = r_array[window_idx]
    k_window = k_array[window_idx]
    known = np.logical_not(r_window == 0)

    relevant_label = 1 if mode == "specificity" else -1

    while np.sum(r_window == relevant_label) < n_min or np.sum(r_window == -relevant_label) < 1:
        if np.sum(window_idx) < window:
            return np.nanmin(u_array)
        else:
            window += 1
            window_idx = np.logical_and(r_array > -2, k_array >= k_array[-1] - window)
            u_window = u_array[window_idx]
            r_window = r_array[window_idx]
            k_window = k_array[window_idx]
            known = np.logical_not(r_window == 0)
    if u_normalization:

        x = k_window
        y = u_window
        uncertainty_mean = LinearRegression().fit(x.reshape(-1, 1), y)
        u_window = (
            u_window
            - uncertainty_mean.predict(x.reshape(-1, 1))
            + uncertainty_mean.predict(np.array([k_window[-1]]).reshape(-1, 1))
        )

    if impute:
        failures = -r_window.copy()
        unknown = np.logical_not(known)
        y = failures[known]
        X = u_window[known].reshape(-1, 1)
        clf = LogisticRegression(penalty=None).fit(X, y)
        probas = clf.predict_proba(u_window[unknown].reshape(-1, 1))
        gammas = []
        for _ in range(n_rep):
            failures[unknown] = np.asarray(probas[:, 1] > np.random.rand(probas.shape[0]), dtype="int") * 2 - 1
            if mode == "sensitivity":
                _, tpr, threshs = roc_curve(failures, u_window, pos_label=1, drop_intermediate=False)
                gamma = np.interp(s_des, tpr + p_rand * (1 - tpr), threshs)
            elif mode == "success":
                success, threshs = success_curve(failures, u_window, pos_label=1)
                gamma = np.interp(s_des, success + p_rand * (1 - success), threshs)
            elif mode == "specificity":
                fpr, _, threshs = roc_curve(failures, u_window, pos_label=1, drop_intermediate=False)
                tnr = 1 - fpr
                gamma = np.interp(s_des, (1 - p_rand) * tnr[::-1], threshs[::-1])
            gammas.append(gamma)
        gamma = np.median(gammas)
    elif mode == "sensitivity":
        _, tpr, threshs = roc_curve(-r_window[known], u_window[known], pos_label=1, drop_intermediate=False)
        gamma = np.interp(s_des, tpr + p_rand * (1 - tpr), threshs)
    elif mode == "success":
        success, threshs = success_curve(-r_window[known], u_window[known], pos_label=1)
        gamma = np.interp(s_des, success + p_rand * (1 - success), threshs)
    elif mode == "specificity":
        fpr, _, threshs = roc_curve(-r_window[known], u_window[known], pos_label=1, drop_intermediate=False)
        tnr = 1 - fpr
        gamma = np.interp(s_des, (1 - p_rand) * tnr[::-1], threshs[::-1], drop_intermediate=False)
    if np.isnan(gamma):
        gamma = np.nanmin(u_window)
    return gamma


def success_curve(y_true, y_score, pos_label=1):
    index_array = np.argsort(y_score)
    y_true_sorted = y_true[index_array]
    y_score_sorted = y_score[index_array]

    positives = y_true_sorted == pos_label
    negatives = y_true_sorted == -pos_label

    success = [1.0]
    tps = [np.sum(positives)]
    fps = [np.sum(negatives)]
    tns = [0]
    fns = [0]
    thresholds = [0.0]

    for i in range(len(y_true)):
        y_t = y_true_sorted[i]
        y_s = y_score_sorted[i]

        tp = tps[-1]
        fp = fps[-1]
        tn = tns[-1]
        fn = fns[-1]

        if y_t == pos_label:
            tp -= 1
            fn += 1
        else:
            fp -= 1
            tn += 1

        if y_s == thresholds[-1]:
            tps[-1] = tp
            fps[-1] = fp
            tns[-1] = tn
            fns[-1] = fn
            success[-1] = 1 - fn / (tp + fp + tn + fn)
        else:
            thresholds.append(float(y_s))
            tps.append(tp)
            fps.append(fp)
            tns.append(tn)
            fns.append(fn)
            success.append(1 - fn / (tp + fp + tn + fn))
    return np.asarray(success)[::-1], np.asarray(thresholds)[::-1]
