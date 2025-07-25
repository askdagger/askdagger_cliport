import numpy as np
import askdagger_cliport


def prioritize_sampling(u, r, k, b=10, lam=0.5):
    u = np.asarray(u, dtype=float) / np.nanmax(u) if np.nanmax(u) != 0 else np.asarray(u, dtype=float)
    r_copy = np.array(r, dtype=float)
    k_copy = np.array(k, dtype=float) / np.max(k) if np.max(k) > 0 else np.asarray(k, dtype=float)

    r_copy[np.asarray(r) == askdagger_cliport.UNKNOWN_RELABELING] = askdagger_cliport.UNKNOWN_ONLINE
    r_copy[np.asarray(r) == askdagger_cliport.UNKNOWN_OFFLINE] = askdagger_cliport.UNKNOWN_ONLINE

    c = lam * u + (1 - lam) * (np.max(k_copy) - k_copy)
    p = 1 - r_copy * (b ** (1 - c) - 1) / (b - 1)
    p[np.isnan(p)] = 1
    return p
