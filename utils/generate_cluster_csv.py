import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


EPS = 1e-8
LOW_VAR_THRESHOLD = 1e-10
HIGH_CORR_THRESHOLD = 0.95


def _shannon_entropy_uint8(img_u8):
    hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _auto_canny(img_u8):
    v = np.median(img_u8)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    return cv2.Canny(img_u8, lower, upper)


def _radial_power_spectrum(img_f):
    f = np.fft.fftshift(np.fft.fft2(img_f))
    p = np.abs(f) ** 2
    h, w = p.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    radial_sum = np.bincount(radius.ravel(), p.ravel())
    radial_count = np.bincount(radius.ravel()) + EPS
    radial = radial_sum / radial_count
    freq = np.arange(len(radial)) / (max(h, w) / 2.0 + EPS)
    return freq[2:], radial[2:]


def _glcm_features(img_u8, levels=32):
    if levels != 256:
        img_q = (img_u8.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)
    else:
        img_q = img_u8

    glcm = graycomatrix(
        img_q,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    feats = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())

    p = glcm / (glcm.sum() + EPS)
    feats["glcm_entropy"] = float(-np.sum(p * np.log(p + EPS)))
    return feats


def _gini_coefficient(arr):
    x = arr.astype(np.float64).ravel()
    x -= x.min()
    s = x.sum()
    if s <= EPS:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float(2.0 * (idx * x).sum() / (n * s) - (n + 1.0) / n)


def _spectral_stats(freq, radial):
    power = np.clip(radial.astype(np.float64), EPS, None)
    freq = np.clip(freq.astype(np.float64), EPS, None)
    centroid = float((freq * power).sum() / (power.sum() + EPS))
    bandwidth = float(np.sqrt(((freq - centroid) ** 2 * power).sum() / (power.sum() + EPS)))
    flatness = float(np.exp(np.mean(np.log(power))) / (np.mean(power) + EPS))
    return centroid, bandwidth, flatness


def _edge_anisotropy(gx, gy):
    ex = np.abs(gx).sum()
    ey = np.abs(gy).sum()
    return float(abs(ex - ey) / (ex + ey + EPS))


def _lbp_features(img_u8, p=8, r=1.0):
    lbp = local_binary_pattern(img_u8, P=p, R=r, method="uniform")
    n_bins = p + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    nz = hist[hist > 0]
    return {
        "lbp_entropy": float(-(nz * np.log2(nz)).sum()),
        "lbp_uniformity": float((hist ** 2).sum()),
        "lbp_maxbin": float(hist.max()),
        "lbp_nonzero_bins": int((hist > 0).sum()),
    }


def _gabor_stats(img_f):
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    lambdas = [4.0, 8.0]
    energies = []
    ori_energy = []
    for theta in thetas:
        ori_sum = 0.0
        for lam in lambdas:
            kernel = cv2.getGaborKernel((21, 21), 2.0, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(img_f, cv2.CV_32F, kernel)
            energy = float(np.mean(np.abs(resp)))
            energies.append(energy)
            ori_sum += energy
        ori_energy.append(ori_sum)

    en = np.array(energies, dtype=np.float32)
    ori = np.array(ori_energy, dtype=np.float32)
    return {
        "gabor_energy_mean": float(en.mean()),
        "gabor_energy_std": float(en.std()),
        "gabor_anisotropy": float((ori.max() - ori.min()) / (ori.max() + ori.min() + EPS)),
    }


def _boxcount_fractal_dimension(edge_bin):
    h, w = edge_bin.shape
    max_pow = int(np.floor(np.log2(min(h, w)))) if min(h, w) > 0 else 1
    sizes = [2 ** k for k in range(1, max_pow)]
    if len(sizes) < 2:
        return 0.0

    counts = []
    for size in sizes:
        sh = (h // size) * size
        sw = (w // size) * size
        cropped = edge_bin[:sh, :sw]
        blocks = cropped.reshape(sh // size, size, sw // size, size)
        counts.append(int((blocks.max(axis=(1, 3)) > 0).sum()))

    x = np.log(1.0 / np.array(sizes, dtype=np.float64))
    y = np.log(np.array(counts, dtype=np.float64) + EPS)
    slope, _ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    return float(slope)


def _mscn_stats(img_f, sigma=7 / 6):
    mu = cv2.GaussianBlur(img_f, (0, 0), sigma)
    mu_sq = cv2.GaussianBlur(img_f * img_f, (0, 0), sigma)
    var = np.clip(mu_sq - mu * mu, 0, None)
    mscn = (img_f - mu) / (np.sqrt(var) + 1.0)
    varc = float(mscn.var())
    kurt = float(ndi.uniform_filter(mscn ** 4, size=7).mean() / (varc + EPS) ** 2 - 3.0)
    pair_h = float((mscn[:, :-1] * mscn[:, 1:]).mean())
    return {
        "mscn_mean": float(mscn.mean()),
        "mscn_var": varc,
        "mscn_kurtosis": kurt,
        "mscn_pair_h": pair_h,
    }


def _multiscale_variance_trend(img_f, sigmas=(1.0, 2.0, 4.0, 8.0)):
    vars_ = []
    for sigma in sigmas:
        vars_.append(float(cv2.GaussianBlur(img_f, (0, 0), sigma).var()))
    x = np.log(np.array(sigmas) + EPS)
    y = np.log(np.array(vars_) + EPS)
    slope, _ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    return {
        "ms_var_slope": float(slope),
        "ms_var_ratio_s1_over_sN": float((vars_[0] + EPS) / (vars_[-1] + EPS)),
    }


def _otsu_stats(img_u8):
    thresh, _ = cv2.threshold(img_u8, 0, 255, cv2.THRESH_OTSU)
    fg = (img_u8 >= thresh).astype(np.uint8)
    hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + EPS)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = np.max((mu_t * omega - mu) ** 2 / (omega * (1 - omega) + EPS))
    return {
        "otsu_thresh": float(thresh),
        "otsu_fg_ratio": float(fg.mean()),
        "otsu_between_var": float(sigma_b2),
    }


def calculate_image_metrics(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    h, w = img.shape
    npix = float(h * w)
    img_u8 = img
    img_f = img.astype(np.float32)
    img01 = img_f / 255.0

    mean = img_f.mean()
    std = img_f.std()
    dynamic_range = float(img_f.max() - img_f.min())
    entropy = _shannon_entropy_uint8(img_u8)

    metrics = {
        "image_name": Path(img_path).name,
        "image_width": w,
        "image_height": h,
        "mean": float(mean),
        "std": float(std),
        "dynamic_range": dynamic_range,
        "rms_contrast": float(std / (mean + EPS)),
        "entropy_global": float(entropy),
    }

    lap_var = cv2.Laplacian(img_u8, cv2.CV_64F).var()
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = (np.arctan2(gy, gx) + np.pi) % (2 * np.pi)
    hist, _ = np.histogram(angle.ravel(), bins=36, range=(0, 2 * np.pi), weights=grad_mag.ravel())
    p = hist / (hist.sum() + EPS)
    metrics.update({
        "laplacian_var": float(lap_var),
        "tenengrad": float((grad_mag ** 2).mean()),
        "grad_mean": float(grad_mag.mean()),
        "grad_std": float(grad_mag.std()),
        "grad_orient_entropy": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
    })

    edges = _auto_canny(img_u8)
    edge_density = float((edges > 0).sum() / (npix + EPS))
    corners = cv2.goodFeaturesToTrack(img_u8, maxCorners=1000, qualityLevel=0.01, minDistance=3)
    corner_cnt = 0 if corners is None else corners.shape[0]

    max_f = cv2.dilate(img_u8, np.ones((3, 3), np.uint8))
    peak_cnt = int(((img_u8 == max_f) & (img_u8 >= (mean + std))).sum())
    ppm = 1e6 / (npix + EPS)
    metrics.update({
        "edge_density": edge_density,
        "corner_density_Mpix": float(corner_cnt * ppm),
        "peak_density_Mpix": float(peak_cnt * ppm),
    })

    metrics.update(_glcm_features(img_u8, levels=32))
    blur_small = cv2.GaussianBlur(img_f, (0, 0), 1.5)
    blur_large = cv2.GaussianBlur(img_f, (0, 0), 7.0)
    metrics["local_contrast_mean"] = float(np.abs(blur_small - blur_large).mean())

    freq, radial = _radial_power_spectrum(img01)
    lo = int(0.1 * len(freq))
    hi = int(0.9 * len(freq))
    if hi > lo + 5:
        x = np.log(freq[lo:hi] + EPS)
        y = np.log(radial[lo:hi] + EPS)
        slope, _ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    else:
        slope = 0.0
    hf_ratio = float((radial[freq >= 0.3].sum() + EPS) / (radial.sum() + EPS))
    metrics.update({
        "psd_log_slope": float(slope),
        "psd_hf_ratio": hf_ratio,
    })

    residual = img_f - cv2.GaussianBlur(img_f, (0, 0), 1.2)
    mad = np.median(np.abs(residual - np.median(residual)))
    illum = cv2.GaussianBlur(img_f, (0, 0), 21.0)
    metrics.update({
        "noise_sigma": float(1.4826 * mad),
        "illum_nonuniform": float(illum.std() / (dynamic_range + EPS)),
    })

    p05 = float(np.percentile(img_f, 5))
    p25 = float(np.percentile(img_f, 25))
    p50 = float(np.percentile(img_f, 50))
    p75 = float(np.percentile(img_f, 75))
    p95 = float(np.percentile(img_f, 95))
    p99 = float(np.percentile(img_f, 99))
    robust_std = float(1.4826 * np.median(np.abs(img_f - p50)) + EPS)
    z = (img_f - mean) / (std + EPS)
    metrics.update({
        "p95_minus_p50_over_robuststd": float((p95 - p50) / (robust_std + EPS)),
        "p99_minus_p50_over_robuststd": float((p99 - p50) / (robust_std + EPS)),
        "skewness": float((z ** 3).mean()),
        "kurtosis": float((z ** 4).mean()) - 3.0,
        "p05": p05,
        "p25": p25,
        "p75": p75,
        "gini_intensity": float(_gini_coefficient(img_f)),
    })

    sobx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    metrics["edge_anisotropy"] = _edge_anisotropy(sobx, soby)
    k3 = np.ones((3, 3), np.uint8)
    band = cv2.dilate((edges > 0).astype(np.uint8), k3, iterations=1) - cv2.erode((edges > 0).astype(np.uint8), k3, iterations=1)
    band = band > 0
    grad_mag2 = np.sqrt(sobx ** 2 + soby ** 2)
    metrics["edge_band_grad_mean"] = float(grad_mag2[band].mean() if band.any() else 0.0)

    metrics.update(_lbp_features(img_u8, p=8, r=1.0))
    metrics.update(_gabor_stats(img_f))

    centroid, bandwidth, flatness = _spectral_stats(freq, radial)
    metrics.update({
        "psd_centroid": centroid,
        "psd_bandwidth": bandwidth,
        "psd_flatness": flatness,
    })

    metrics["fractal_dim_edges"] = _boxcount_fractal_dimension((edges > 0).astype(np.uint8))

    metrics.update(_mscn_stats(img_f))
    metrics.update(_multiscale_variance_trend(img_f))
    metrics.update(_otsu_stats(img_u8))
    return metrics


def build_metrics_dataframe(dataset_root):
    img_dir = Path(dataset_root) / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png")
    if not image_paths:
        raise RuntimeError(f"No PNG images found in: {img_dir}")

    rows = []
    print(f"Processing {len(image_paths)} images in {img_dir}")
    for img_path in tqdm(image_paths):
        metrics = calculate_image_metrics(img_path)
        metrics["dataset"] = Path(dataset_root).name
        rows.append(metrics)

    return pd.DataFrame(rows)


def _prepare_features(df):
    exclude = {"image_name", "dataset", "cluster_id"}
    feature_cols = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for clustering.")

    x = df[feature_cols].to_numpy(dtype=np.float32)
    if np.isnan(x).any():
        medians = np.nanmedian(x, axis=0)
        idxs = np.where(np.isnan(x))
        x[idxs] = np.take(medians, idxs[1])
    return x, feature_cols


def _standardize(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)


def _filter_features(x, feature_cols, corr_threshold):
    var = np.var(x, axis=0)
    keep = var > LOW_VAR_THRESHOLD
    x = x[:, keep]
    feature_cols = [name for name, ok in zip(feature_cols, keep) if ok]
    var = var[keep]
    if len(feature_cols) <= 1:
        return x, feature_cols

    z = _standardize(x)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            value = abs(corr[i, j])
            if value >= corr_threshold:
                pairs.append((i, j, value))
    pairs.sort(key=lambda item: -item[2])

    keep = np.ones(len(feature_cols), dtype=bool)
    for i, j, _ in pairs:
        if not (keep[i] and keep[j]):
            continue
        if var[i] >= var[j]:
            keep[j] = False
        else:
            keep[i] = False

    return x[:, keep], [name for name, ok in zip(feature_cols, keep) if ok]


def _pca(x_scaled):
    if 0.9 >= 1.0:
        return x_scaled

    pca_full = PCA(random_state=1)
    pca_full.fit(x_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, 0.9) + 1)
    n_comp = min(n_comp, 20)
    n_comp = max(2, n_comp)
    n_comp = min(n_comp, x_scaled.shape[1], x_scaled.shape[0])
    print(f"PCA components: {n_comp}, cumulative variance: {cum[n_comp - 1]:.4f}")
    return PCA(n_components=n_comp, random_state=1).fit_transform(x_scaled)


def _kmeans(x):
    if 256 < 1:
        raise ValueError("k must be >= 1.")
    if 256 > x.shape[0]:
        raise ValueError(f"k=256 is larger than number of samples={x.shape[0]}.")

    kmeans = KMeans(
        n_clusters=256,
        random_state=1,
        n_init=1,
        max_iter=300,
    )
    labels = kmeans.fit_predict(x)
    print(f"KMeans inertia: {kmeans.inertia_:.4f}")
    return labels


def _rank_features(x_scaled, labels):
    valid = np.nanvar(x_scaled, axis=0) > LOW_VAR_THRESHOLD
    scores = np.full(x_scaled.shape[1], -np.inf, dtype=np.float64)
    if valid.any() and len(np.unique(labels)) > 1:
        f_values, _ = f_classif(x_scaled[:, valid], labels)
        mi_values = mutual_info_classif(x_scaled[:, valid], labels, random_state=1)
        f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
        mi_values = np.nan_to_num(mi_values, nan=0.0, posinf=0.0, neginf=0.0)
        scores[valid] = mi_values + 1e-6 * f_values
    return scores


def cluster_metrics_dataframe(df, output_csv):
    x, feature_cols = _prepare_features(df)
    x, feature_cols = _filter_features(x, feature_cols, HIGH_CORR_THRESHOLD)
    x_scaled = _standardize(x)
    x_embed = _pca(x_scaled)
    labels = _kmeans(x_embed.astype(np.float64))

    ranks = _rank_features(x_scaled, labels)
    top_idx = np.argsort(-ranks)[:min(32, len(feature_cols))]
    top_features = [feature_cols[i] for i in top_idx]
    print(f"Refitting with top-{len(top_features)} features.")
    x_top = df[top_features].to_numpy(dtype=np.float32)
    if np.isnan(x_top).any():
        medians = np.nanmedian(x_top, axis=0)
        idxs = np.where(np.isnan(x_top))
        x_top[idxs] = np.take(medians, idxs[1])
    x_top, _ = _filter_features(x_top, top_features, HIGH_CORR_THRESHOLD)
    x_top_scaled = _standardize(x_top)
    x_top_embed = _pca(x_top_scaled)
    labels = _kmeans(x_top_embed.astype(np.float64))

    out = df.copy()
    out["cluster_id"] = labels.astype(int)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved clustered CSV to: {output_csv}")


def main():
    this_dir = Path(__file__).resolve().parent
    cluster_metrics_dataframe(
        build_metrics_dataframe(this_dir.parent / "datasets" / "SIRST3"),
        output_csv=this_dir / "ir_image_metrics_with_clusters.csv",
    )


if __name__ == "__main__":
    main()
