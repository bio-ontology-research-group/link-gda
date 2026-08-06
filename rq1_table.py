"""RQ1 results table: every method in both settings.

Uncalibrated setting pairs a raw-selected checkpoint with raw metrics; calibrated setting
pairs a calibrated-selected checkpoint with calibrated metrics, so each column is
internally coherent. Symbolic baselines have no training, so both settings come from the
same score files.

Calibration is leave-one-out per-gene z-scoring, applied identically to every method.
Ranks use the optimistic convention. Deviations are sample standard deviations over folds.

    python rq1_table.py --spec specs.tsv

specs.tsv columns: label, kind (learned|symbolic), raw_template, cal_template
Templates take {f} for the fold. For symbolic methods both templates are the same file.
"""
import csv
import math

import click as ck
import numpy as np


def load(path):
    scores, idx = [], []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            idx.append(int(parts[2]))
            scores.append(np.fromiter((float(x) for x in parts[3:]), dtype=np.float64))
    return np.vstack(scores), np.array(idx)


def calibrate(scores):
    n = scores.shape[0]
    mean = (scores.sum(axis=0, keepdims=True) - scores) / (n - 1)
    var = ((scores ** 2).sum(axis=0, keepdims=True) - scores ** 2) / (n - 1) - mean ** 2
    return (scores - mean) / (np.sqrt(np.clip(var, 0, None)) + 1e-12)


def metrics(scores, idx):
    ranks = np.array([1 + int((scores[i] > scores[i, idx[i]]).sum()) for i in range(scores.shape[0])])
    return {"mr": ranks.mean(), "mrr": (1 / ranks).mean(),
            "h1": (ranks <= 1).mean(), "h10": (ranks <= 10).mean(),
            "h100": (ranks <= 100).mean()}


def over_folds(template, apply_calibration):
    import os
    rows = []
    for fold in range(10):
        path = template.format(f=fold)
        if not os.path.exists(path):
            continue
        scores, idx = load(path)
        rows.append(metrics(calibrate(scores) if apply_calibration else scores, idx))
    return rows


def summarise(rows, key):
    values = np.array([r[key] for r in rows])
    return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0


def paired_t(a, b):
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    n = len(diff)
    if n < 2:
        return float("nan")
    spread = diff.std(ddof=1)
    if spread == 0:
        return 1.0 if diff.mean() == 0 else 0.0
    t = diff.mean() / (spread / math.sqrt(n))
    df = n - 1
    x = df / (df + t * t)

    def betacf(aa, bb, xx):
        tiny = 1e-300
        qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 201):
            m2 = 2 * m
            num = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + num * d
            c = 1.0 + num / c
            if abs(d) < tiny:
                d = tiny
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            num = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + num * d
            c = 1.0 + num / c
            if abs(d) < tiny:
                d = tiny
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-16:
                break
        return h

    def betai(aa, bb, xx):
        if xx <= 0.0:
            return 0.0
        if xx >= 1.0:
            return 1.0
        lbeta = math.lgamma(aa + bb) - math.lgamma(aa) - math.lgamma(bb)
        front = math.exp(lbeta + aa * math.log(xx) + bb * math.log(1.0 - xx))
        if xx < (aa + 1.0) / (aa + bb + 2.0):
            return front * betacf(aa, bb, xx) / aa
        return 1.0 - front * betacf(bb, aa, 1.0 - xx) / bb

    return betai(df / 2.0, 0.5, x)


@ck.command()
@ck.option("--spec", required=True, help="TSV of label, kind, raw_template, cal_template")
@ck.option("--reference", default=None, help="Label to run paired fold-level tests against")
def main(spec, reference):
    specs = []
    with open(spec) as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            specs.append(row)

    results = {}
    print(f"{'method':<20}{'setting':<14}{'MR':>17}{'MRR':>10}{'H@1':>9}{'H@10':>9}{'H@100':>9}")
    for label, kind, raw_tmpl, cal_tmpl in specs:
        for setting, tmpl, cal in (("uncalibrated", raw_tmpl, False), ("calibrated", cal_tmpl, True)):
            rows = over_folds(tmpl, cal)
            if not rows:
                print(f"{label:<20}{setting:<14}   no files")
                continue
            results[(label, setting)] = rows
            mr, mr_sd = summarise(rows, "mr")
            line = f"{label:<20}{setting:<14}{mr:11.2f}±{mr_sd:<5.2f}"
            for key, width in (("mrr", 10), ("h1", 9), ("h10", 9), ("h100", 9)):
                line += f"{summarise(rows, key)[0]:>{width}.4f}"
            print(line + f"   ({len(rows)} folds)")

    if reference:
        print(f"\npaired t-test over folds, mean rank, each method vs {reference}")
        for setting in ("uncalibrated", "calibrated"):
            ref = results.get((reference, setting))
            if not ref:
                continue
            print(f"  {setting}")
            for (label, s), rows in results.items():
                if s != setting or label == reference or len(rows) != len(ref):
                    continue
                a = [r["mr"] for r in rows]
                b = [r["mr"] for r in ref]
                wins = sum(1 for x, y in zip(a, b) if x < y)
                print(f"    {label:<18} delta {np.mean(a) - np.mean(b):+9.2f}   "
                      f"wins {wins}/{len(a)}   p = {paired_t(a, b):.4f}")


if __name__ == "__main__":
    main()
