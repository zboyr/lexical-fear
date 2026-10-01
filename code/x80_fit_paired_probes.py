"""x80 Stages 1-2: paired layer-wise linear probes (R = explicit refusal,
H = comply2 compliance) on Qwen3.5-4B final-prompt-token states, direction
construction, and the representation gate G2.

Per outer seed: for every non-embedding state, fit two L2 logistic heads on
the same inner-train standardized matrix with fractional-label BCE (the
normalized binomial NLL; equal ten draws per prompt, count-weighted form
stored alongside), select each head's L2 on inner-validation NLL, report
deviance skill / Brier / calibration / Spearman / AUROC(count>=5) and a
label-permutation null. Select ONE common state by max over states of
min(skill_R, skill_H) on validation, subject to both skills positive and
above their permutation 95th percentiles. Outer-test metrics are computed
only at the selected state, after selection. Then build q_R, q_H,
contrast, refusal_residual, permutation and random directions at that
state, with prompt-bootstrap angular stability.

Inputs: the x63 cache hidden_states.pt (list of L [30000, d] fp16 tensors),
x80_labels.npz (cache rows, R counts, H counts), x66 split index files.
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

L2_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]  # inherited from x71 (pre-registered)
L2_GRID_EXT = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]  # sensitivity: the x71 grid saturated at 1e-3
MAX_ITER = 80
N_DRAWS = 10


def fit_logistic(x, y, l2, max_iter=MAX_ITER):
    w = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros((), device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y)
        if l2:
            loss = loss + 0.5 * l2 * w.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def nll(logits, y):
    return float(F.binary_cross_entropy_with_logits(logits.double(), y.double()))


def calib(logits, y):
    """Calibration intercept/slope: logistic regression of y on the logit."""
    a = torch.zeros((), device=logits.device, requires_grad=True)
    c = torch.ones((), device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([a, c], max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(a + c * logits, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a), float(c)


def metrics(logits, y, counts, nll0):
    p = torch.sigmoid(logits)
    ln = nll(logits, y)
    a, c = calib(logits, y)
    yn, pn = y.cpu().numpy(), p.cpu().numpy()
    maj = counts >= 5
    auc = float(roc_auc_score(maj, pn)) if 0 < maj.sum() < len(maj) else None
    return {"n": int(len(y)), "nll": ln, "nll_intercept": nll0, "deviance_skill": 1.0 - ln / nll0,
            "count_weighted_nll": ln * N_DRAWS,
            "brier": float(((p - y) ** 2).mean()), "calib_intercept": a, "calib_slope": c,
            "spearman": float(spearmanr(pn, yn).statistic), "auroc_count_ge5": auc,
            "mean_target": float(y.mean()), "mean_pred": float(p.mean())}


def intercept_nll(y_train, y_eval):
    m = float(y_train.mean())
    return float(-(y_eval.double() * math.log(m) + (1 - y_eval.double()) * math.log(1 - m)).mean())


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--indices-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--l2-ext", action="store_true", help="use L2_GRID_EXT (sensitivity run)")
    args = ap.parse_args()
    global L2_GRID
    if args.l2_ext:
        L2_GRID = L2_GRID_EXT
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    lab = np.load(args.labels)
    rows, Rc, Hc = lab["rows"], lab["R_count"], lab["H_count"]
    pos = {int(r): k for k, r in enumerate(rows)}
    tr = np.array(json.load(open(f"{args.indices_dir}/qwen_seed{args.seed}_train_indices.json")))
    te = np.array(json.load(open(f"{args.indices_dir}/qwen_seed{args.seed}_test_indices.json")))
    src = np.zeros(30000, dtype=np.int8); src[:15000] = 1
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=42, stratify=src[tr])  # x79 convention
    sets = {"train": tr[itr], "val": tr[iva], "test": te}
    sets = {k: np.array([r for r in v if r < 15000]) for k, v in sets.items()}
    for k, v in sets.items():
        assert all(int(r) in pos for r in v), f"{k}: non-retained harmful row in split"
    cnt = {k: {"R": Rc[[pos[int(r)] for r in v]], "H": Hc[[pos[int(r)] for r in v]]} for k, v in sets.items()}
    y = {k: {t: torch.tensor(cnt[k][t] / N_DRAWS, dtype=torch.float32, device=dev) for t in "RH"} for k in sets}
    print({k: len(v) for k, v in sets.items()}, flush=True)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    states = cache["tbg_states"]
    L = len(states)
    print(f"states={L} d={states[0].shape[1]} load {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(20260822 + args.seed)
    perm_idx = [rng.permutation(len(sets["train"])) for _ in range(args.n_perm)]

    per_state = {}
    fits = {}
    for s in range(1, L):
        X = {k: states[s][torch.from_numpy(v)].to(dev).float() for k, v in sets.items()}
        mean, std = X["train"].mean(0), X["train"].std(0, unbiased=False).clamp_min(1e-4)
        Z = {k: (X[k] - mean) / std for k in X}
        rec = {"state": s}
        fits[s] = {"mean": mean.cpu(), "std": std.cpu()}
        for t in "RH":
            nll0 = intercept_nll(y["train"][t], y["val"][t])
            best = None
            for l2 in L2_GRID:
                w, b = fit_logistic(Z["train"], y["train"][t], l2)
                m = metrics(Z["val"] @ w + b, y["val"][t], cnt["val"][t], nll0)
                if best is None or m["nll"] < best[1]["nll"]:
                    best = (l2, m, w, b)
            l2, m, w, b = best
            # permutation null at the selected l2: shuffle prompts' labels on train, score val
            null = []
            for pi in perm_idx:
                wp, bp = fit_logistic(Z["train"], y["train"][t][torch.from_numpy(pi).to(dev)], l2)
                lp = Z["val"] @ wp + bp
                null.append(1.0 - nll(lp, y["val"][t]) / nll0)
            m.update({"l2": l2, "coef_norm_std": float(w.norm()), "coef_norm_raw": float((w / std).norm()),
                      "perm_null_skill": null, "perm_null_p95": float(np.percentile(null, 95)),
                      "beats_null": bool(m["deviance_skill"] > np.percentile(null, 95) and m["deviance_skill"] > 0)})
            rec[t] = m
            fits[s][t] = (w.cpu(), b.cpu())
        qR, qH = fits[s]["R"][0] / std.cpu(), fits[s]["H"][0] / std.cpu()
        rec["cos_qR_qH"] = cos(qR, qH)
        rec["min_skill"] = min(rec["R"]["deviance_skill"], rec["H"]["deviance_skill"])
        rec["eligible"] = bool(rec["R"]["beats_null"] and rec["H"]["beats_null"])
        per_state[s] = rec
        print(f"state {s:2d} skillR {rec['R']['deviance_skill']:.3f} skillH {rec['H']['deviance_skill']:.3f} "
              f"cos {rec['cos_qR_qH']:+.3f} elig {rec['eligible']} l2 {rec['R']['l2']:g}/{rec['H']['l2']:g} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        del X, Z

    elig = [s for s in per_state if per_state[s]["eligible"]]
    sel = max(elig, key=lambda s: per_state[s]["min_skill"]) if elig else None
    out = {"seed": args.seed, "sizes": {k: int(len(v)) for k, v in sets.items()}, "l2_grid": L2_GRID,
           "per_state": per_state, "eligible_states": elig, "selected_state": sel,
           "H_caveat": "H is comply2 substantive compliance, not judged harm"}
    if sel is None:
        out["G2"] = False; out["G2_reason"] = "no state where both heads beat their permutation nulls"
        json.dump(out, open(f"{args.out_dir}/x80_probes_s{args.seed}.json", "w"), indent=1)
        return

    # ---- Stage 2 at the selected state: directions, test metrics (sealed until here), bootstrap stability
    s = sel
    X = {k: states[s][torch.from_numpy(v)].to(dev).float() for k, v in sets.items()}
    mean, std = fits[s]["mean"].to(dev), fits[s]["std"].to(dev)
    Z = {k: (X[k] - mean) / std for k in X}
    test = {}
    for t in "RH":
        w, b = (v.to(dev) for v in fits[s][t])
        nll0 = intercept_nll(y["train"][t], y["test"][t])
        test[t] = metrics(Z["test"] @ w + b, y["test"][t], cnt["test"][t], nll0)
    wR, bR = (v.to(dev) for v in fits[s]["R"]); wH, bH = (v.to(dev) for v in fits[s]["H"])
    qR = wR / std; qH = wH / std
    uR, uH = qR / qR.norm(), qH / qH.norm()
    contrast = uR - uH; contrast = contrast / contrast.norm()
    resid = uR - torch.dot(uR, uH) * uH; resid = resid / resid.norm()
    g = torch.Generator(device="cpu").manual_seed(20260822 + args.seed)
    rand = torch.randn(uR.shape[0], generator=g).to(dev); rand = rand / rand.norm()
    # permutation contrast: paired label permutation (same permutation for R and H)
    pi = torch.from_numpy(perm_idx[0]).to(dev)
    wRp, _ = fit_logistic(Z["train"], y["train"]["R"][pi], per_state[s]["R"]["l2"])
    wHp, _ = fit_logistic(Z["train"], y["train"]["H"][pi], per_state[s]["H"]["l2"])
    up = wRp / std; up = up / up.norm(); vp = wHp / std; vp = vp / vp.norm()
    permc = up - vp; permc = permc / permc.norm()
    dirs = {"refusal": uR, "harm_suppress": -uH, "contrast": contrast, "refusal_residual": resid,
            "permutation": permc, "random": rand}
    names = list(dirs)
    cosm = [[cos(dirs[a], dirs[b]) for b in names] for a in names]

    boots = {"refusal": [], "harm_suppress": [], "contrast": [], "refusal_residual": []}
    n = Z["train"].shape[0]
    for _ in range(args.n_boot):
        bi = torch.from_numpy(rng.integers(0, n, n)).to(dev)
        Zb = Z["train"][bi]
        wRb, _ = fit_logistic(Zb, y["train"]["R"][bi], per_state[s]["R"]["l2"])
        wHb, _ = fit_logistic(Zb, y["train"]["H"][bi], per_state[s]["H"]["l2"])
        a = wRb / std; a = a / a.norm(); c = wHb / std; c = c / c.norm()
        cb = a - c; rb = a - torch.dot(a, c) * c
        boots["refusal"].append(cos(a, uR)); boots["harm_suppress"].append(cos(-c, -uH))
        boots["contrast"].append(cos(cb / cb.norm(), contrast)); boots["refusal_residual"].append(cos(rb / rb.norm(), resid))
    stab = {}
    for k, v in boots.items():
        v = np.array(v); ang = np.degrees(np.arccos(np.clip(v, -1, 1)))
        stab[k] = {"cos_median": float(np.median(v)), "cos_p10": float(np.percentile(v, 10)),
                   "angle_p90_deg": float(np.percentile(ang, 90)), "n_boot": args.n_boot}
    g2 = (stab["contrast"]["cos_median"] >= 0.5 and stab["contrast"]["angle_p90_deg"] < 45
          and abs(cos(contrast, uR)) < 0.995 and abs(cos(contrast, -uH)) < 0.995)
    out.update({"test_at_selected": test, "direction_names": names, "cosine_matrix": cosm,
                "cos_qR_minus_qH_raw": cos(qR, -qH), "angle_qR_qH_deg": float(np.degrees(np.arccos(np.clip(cos(qR, qH), -1, 1)))),
                "bootstrap_stability": stab, "G2": bool(g2),
                "G2_parts": {"both_heads_pass": True, "contrast_stable": bool(stab["contrast"]["cos_median"] >= 0.5 and stab["contrast"]["angle_p90_deg"] < 45),
                             "contrast_distinct": bool(abs(cos(contrast, uR)) < 0.995 and abs(cos(contrast, -uH)) < 0.995)},
                "elapsed_s": time.time() - t0})
    json.dump(out, open(f"{args.out_dir}/x80_probes_s{args.seed}.json", "w"), indent=1)
    torch.save({"schema_version": 1, "experiment": "x80", "seed": args.seed, "state_index": s, "block_index": s - 1,
                "mean": mean.cpu(), "std": std.cpu(),
                "R": {"w_standardized": wR.cpu(), "bias": bR.cpu(), "l2": per_state[s]["R"]["l2"], "orientation": "positive = refusal (= -v6)"},
                "H": {"w_standardized": wH.cpu(), "bias": bH.cpu(), "l2": per_state[s]["H"]["l2"], "orientation": "positive = comply2 compliance (NOT judged harm)"},
                "directions_unit_raw": {k: v.cpu() for k, v in dirs.items()}},
               f"{args.out_dir}/x80_paired_probe_s{args.seed}.pt")
    print(json.dumps({"selected_state": s, "G2": g2, "test": {t: round(test[t]["deviance_skill"], 3) for t in test},
                      "cos_qR_-qH": round(cos(qR, -qH), 3), "stab_contrast": stab["contrast"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
