import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "reproducibility" / "results" / "stepwise-bias-report-v4.json"
DEFAULT_OUT_DIR = REPO_ROOT / "reproducibility" / "tables"


def fmt_pair(st):
    return f"{st['h10']:.2f} / {st['mr']:.1f}"


def fmt_h1_h10_mr(st):
    return f"{st['h1']:.2f} / {st['h10']:.2f} / {st['mr']:.1f}"


def main():
    ap = argparse.ArgumentParser(description="Generate LaTeX tables for the stepwise bias decomposition and corrections (Table 9 and Suppl. Table S13).")
    ap.add_argument("--report-json", type=Path, default=DEFAULT_REPORT, help="Path to stepwise-bias-report-v4.json")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for generated .tex tables")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUT_MAIN = args.out_dir / "revised-bias-stepwise.tex"
    OUT_SUPP = args.out_dir / "revised-bias-baselines-full.tex"
    d = json.load(open(args.report_json))

    panel_a = [
        ("LinkGDA-U1 (PFS)", r"\method{}-pfs (U1, untyped neg., disease-bridge)"),
        ("LinkGDA-U1 (PF, No Expr)", r"\method{}-pf (U1, no expression hubs)"),
        ("LinkGDA-U1 (Retrained No-Bridge, Raw-Head)", r"\method{}-pfs (U1, retrained no disease bridges)"),
        ("LinkGDA-T1 (Typed Neg)", r"\method{}-pfs (T1, candidate-pool neg., disease-bridge)"),
        ("LinkGDA-T1 (Retrained No-Bridge, Raw-Head)", r"\method{}-pfs (T1, retrained no disease bridges)"),
        ("LinkGDA-Ppool (Profile Hard-Neg)", r"\method{}-pfs (P-pool, profile hard-neg., disease-bridge)"),
        ("LinkGDA-Ppool (Retrained No-Bridge)", r"\method{}-pfs (P-pool, retrained no disease bridges)"),
        ("ConvKB-D (PFS)", r"ConvKB-D (PFS, disease-bridge)"),
        ("ConvKB-D (Retrained No-Bridge, Raw-MR)", r"ConvKB-D (PFS, no-bridge retraining)"),
        ("INDIGENA (PFS)", r"INDIGENA (PFS, disease-bridge)"),
        ("INDIGENA (Retrained No-Bridge, Raw-MR)", r"INDIGENA (PFS, no-bridge retraining)"),
        ("ULTRA (Zero-Shot BMA)", r"ULTRA, zero-shot BMA"),
        ("ULTRA (Control Mean-Pool)", r"ULTRA, disease-bridge, mean pool"),
        ("ULTRA (No Disease Bridges)", r"ULTRA, disease-edge ablation (fixed weights)"),
        ("ULTRA (No Expression Hubs)", r"ULTRA, expression-edge ablation (fixed weights)"),
        ("ULTRA (Direct Overlap Masked)", r"ULTRA, overlapping score entries omitted"),
        ("UltraQuery-Matched (original_pool_adapted)", r"UltraQuery, disease-bridge, Pool, adapted"),
        ("UltraQuery-Matched (augmented_and_adapted)", r"UltraQuery, disease-bridge + EQ, AND, adapted"),
        ("PhenoQuery (bridge_pool)", r"PhenoQuery, disease-bridge (\texttt{bridge\_pool})"),
        ("PhenoQuery (plain_pool)", r"PhenoQuery, no disease bridges (\texttt{plain\_pool})"),
        ("PhenoQuery (plain_and)", r"PhenoQuery, no disease bridges (\texttt{plain\_and})"),
        ("PhenoQuery (entity_and)", r"PhenoQuery, no disease + EQ entities (\texttt{entity\_and})"),
        ("PhenoQuery (typed_pool)", r"PhenoQuery, no disease + EQ-typed (\texttt{typed\_pool})"),
        ("PhenoQuery (typed_and)", r"PhenoQuery, no disease + EQ-typed (\texttt{typed\_and})"),
        ("PhenoQuery (structure_and)", r"PhenoQuery, no disease + full EQ (\texttt{structure\_and})"),
    ]

    panel_b = [
        ("Phrank-HPO (disease_raw)", r"Phrank, human-only UPheno disease-max (\texttt{disease\_raw})"),
        ("Phrank-HPO (direct_raw)", r"Phrank, human-only UPheno direct union (\texttt{direct\_raw})"),
        ("Phrank-UPheno (Human+MGI Disease Raw)", r"Phrank, refitted UPheno H+MGI disease-max"),
        ("Phrank-UPheno (Human+MGI Direct Raw)", r"Phrank, refitted UPheno H+MGI direct"),
        ("Phrank-UPheno (MGI-Only Raw)", r"Phrank, refitted UPheno MGI-only"),
        ("PhenoRank (Static 2016 1000-Sim TB)", r"PhenoRank, 2016 resources, human refit, 1,000-sim.\ $p$-val.\ (TB)"),
        ("PhenoRank-UPheno (HM-Add+PPI 1000-Sim TB)", r"PhenoRank, refitted UPheno H+M+PPI, 1,000-sim.\ $p$-val.\ (TB)"),
        ("PhenoRank-UPheno (HM-Add+PPI 1000-Sim)", r"PhenoRank, refitted UPheno H+M+PPI, 1,000-sim.\ $p$-val.\ (no TB)"),
        ("PhenoRank-UPheno (HM-Add+PPI NoSim)", r"PhenoRank, refitted UPheno H+M+PPI, continuous RWR (no sim.)"),
        ("PhenoRank-UPheno (HM-Bal+PPI 1000-Sim TB)", r"PhenoRank, refitted UPheno H+M-bal.+PPI, 1,000-sim.\ (TB)"),
        ("PhenoRank-UPheno (HM-Bal+PPI NoSim)", r"PhenoRank, refitted UPheno H+M-bal.+PPI, continuous RWR"),
        ("PhenoRank-UPheno (Mouse+PPI 1000-Sim TB)", r"PhenoRank, refitted UPheno Mouse+PPI, 1,000-sim.\ (TB)"),
        ("PhenoRank-UPheno (Mouse+PPI NoSim)", r"PhenoRank, refitted UPheno Mouse+PPI, continuous RWR"),
        ("PhenoRank-UPheno (Mouse-NoPPI 1000-Sim TB)", r"PhenoRank, refitted UPheno Mouse no-PPI, 1,000-sim.\ (TB)"),
        ("PhenoRank-UPheno (Mouse-NoPPI NoSim)", r"PhenoRank, refitted UPheno Mouse no-PPI, continuous RWR"),
        ("Resnik-BMA", r"Resnik-BMA (UPheno MGI)"),
        ("Lin-BMA", r"Lin-BMA (UPheno MGI)"),
        ("SimGIC", r"SimGIC (UPheno MGI)"),
        ("Exomiser-Phive", r"Exomiser-Phive (MGI mouse)"),
    ]

    lines = [
        r"\begin{table*}[tb]",
        r"\centering\scriptsize",
        r"\renewcommand{\arraystretch}{0.86}",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\caption{Scoring adjustments and evaluation cohorts for learned graph models (Panel A) and phenotype-based baselines (Panel B) over all ten folds (13,625 candidate genes). Each cell reports $\mathrm{H@10~(\%)}~/~\mathrm{MR}$. Step~0 (S0) uses raw scores on the full benchmark ($n=6{,}571$ pairs). Step~1 (S1) applies leave-one-disease-out per-gene reference calibration. Step~2 (S2) applies within-query status-stratum standardization (\texttt{Status-Z}), equalizing the mean and variance of seen ($3{,}779.2$) and unseen ($9{,}845.8$) candidate pools per query. Step~3 (S3) restricts evaluation to the zero-direct-overlap cohort ($J_{\mathrm{same}}=0$, $n=4{,}431$ pairs: $945$ seen pairs with zero shared training phenotypes plus $3{,}486$ unseen pairs). Step~4 (S4) additionally deduplicates identical phenotype sets ($P_d=P_{d'}$) in the calibration panel, excludes the 166 exact training-profile matches remaining after S3 (210 in the full benchmark) ($J_{\mathrm{any}}=1$), filters co-causal genes of multi-gene diseases, and weights unique disease profiles equally ($n=4{,}011$ distinct phenotype profiles). The final column reports within-status calibrated mean ranks on the full S1 cohort (Seen out of $3{,}779.2$; Unseen out of $9{,}845.8$).}",
        r"\label{tab:bias_stepwise}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model configuration and intervention & S0: Raw & S1: +Per-gene Cal. & S2: +Status-Z & S3: +$J_{\mathrm{same}}{=}0$ & S4: Combined & Within-status MR \\",
        r" & H@10 / MR & H@10 / MR & H@10 / MR & H@10 / MR & H@10 / MR & Seen / Unseen \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\emph{Panel A: Learned knowledge-graph embedding and relational graph foundation models}} \\",
    ]

    for key, label in panel_a:
        m = d[key]
        ws_pair = f"{m['S2_WithinStatus_Seen']['mr']:.1f} / {m['S2_WithinStatus_Unseen']['mr']:.1f}"
        lines.append(
            f"{label} & {fmt_pair(m['S0_Raw_All'])} & {fmt_pair(m['S1_Cal_All'])} & "
            f"{fmt_pair(m['S2_StatusZ_All'])} & {fmt_pair(m['S3_ZeroDirectOverlap_StatusZ'])} & "
            f"{fmt_pair(m['S4_Cumulative_All4'])} & {ws_pair} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{l}{\emph{Panel B: External ranking methods and non-trained semantic-similarity baselines}} \\",
    ])

    for key, label in panel_b:
        m = d[key]
        ws_pair = f"{m['S2_WithinStatus_Seen']['mr']:.1f} / {m['S2_WithinStatus_Unseen']['mr']:.1f}"
        lines.append(
            f"{label} & {fmt_pair(m['S0_Raw_All'])} & {fmt_pair(m['S1_Cal_All'])} & "
            f"{fmt_pair(m['S2_StatusZ_All'])} & {fmt_pair(m['S3_ZeroDirectOverlap_StatusZ'])} & "
            f"{fmt_pair(m['S4_Cumulative_All4'])} & {ws_pair} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    OUT_MAIN.write_text("\n".join(lines) + "\n")

    # Supplementary table with Seen vs Unseen breakdown at S1 and S2, plus S3 Cal vs S3 StatusZ and S4 Isolated on S1
    supp_lines = [r"\begin{landscape}"]
    for panel, reference, title in [
        (panel_a, 'tab:bias_baselines_full', 'Learned graph models'),
        (panel_b, 'tab:bias_baselines_external', 'Phenotype and network baselines')]:
        supp_lines.extend([
            r"\begin{table*}[p]",
            r"\centering\footnotesize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{0.86}",
            r"\caption{"+title+r": training-status and phenotype-overlap diagnostics (ten folds; 13,625 candidates). Cells give H@1 (\%)/H@10 (\%)/MR. S1 uses per-gene calibration; S2 additionally standardizes candidate-status groups. Seen and unseen strata contain 3,085 and 3,486 associations. S3 restricts evaluation to 4,431 zero-overlap pairs under S1 (Cal.) or S2 (Status-Z) scoring. S4-on-S1 combines reference deduplication, training-profile exclusion, co-causal filtering and equal-profile weighting on S1 scores (5,878 profiles).}",
            r"\label{"+reference+'}',
            r"\begin{adjustbox}{max width=\linewidth}",
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.30\linewidth}ccccccc}",
            r"\toprule",
            r"Method variant & S1 Seen & S1 Unseen & S2 Seen & S2 Unseen & S3: Cal. & S3: Status-Z & S4 on S1 \\",
            r" & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR \\",
            r"\midrule",
        ])
        for key, label in panel:
            m = d[key]
            supp_lines.append(
                f"{label} & {fmt_h1_h10_mr(m['S1_Cal_Seen'])} & {fmt_h1_h10_mr(m['S1_Cal_Unseen'])} & "
                f"{fmt_h1_h10_mr(m['S2_StatusZ_Seen'])} & {fmt_h1_h10_mr(m['S2_StatusZ_Unseen'])} & "
                f"{fmt_h1_h10_mr(m['S3_ZeroDirectOverlap_Cal'])} & {fmt_h1_h10_mr(m['S3_ZeroDirectOverlap_StatusZ'])} & "
                f"{fmt_h1_h10_mr(m['S4_Isolated_Dedup_Filt_EqD_on_S1'])} " + r"\\"
            )
        supp_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
                           r"\end{table*}", r"\clearpage"])
    supp_lines.append(r"\end{landscape}")
    OUT_SUPP.write_text("\n".join(supp_lines) + "\n")
    print("Generated both LaTeX tables.")


if __name__ == "__main__":
    main()
