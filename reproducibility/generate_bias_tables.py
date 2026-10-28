import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "reproducibility" / "results" / "stepwise-bias-report-v3.json"
DEFAULT_OUT_DIR = REPO_ROOT / "reproducibility" / "tables"


def fmt_pair(st):
    return f"{st['h10']:.2f} / {st['mr']:.1f}"


def fmt_h1_h10_mr(st):
    return f"{st['h1']:.2f} / {st['h10']:.2f} / {st['mr']:.1f}"


def main():
    ap = argparse.ArgumentParser(description="Generate LaTeX tables for the stepwise bias decomposition and corrections (Table 9 and Suppl. Table S13).")
    ap.add_argument("--report-json", type=Path, default=DEFAULT_REPORT, help="Path to stepwise-bias-report-v3.json")
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
        ("ConvKB-D (Retrained No-Bridge, Raw-MR)", r"ConvKB-D (PFS, retrained no disease bridges)"),
        ("INDIGENA (PFS)", r"INDIGENA (PFS, disease-bridge)"),
        ("INDIGENA (Retrained No-Bridge, Raw-MR)", r"INDIGENA (PFS, retrained no disease bridges)"),
        ("ULTRA (Zero-Shot BMA)", r"ULTRA, zero-shot BMA"),
        ("ULTRA (Control Mean-Pool)", r"ULTRA, disease-bridge, mean pool"),
        ("ULTRA (No Disease Bridges)", r"ULTRA, no disease bridges (Bias~1 removed)"),
        ("ULTRA (No Expression Hubs)", r"ULTRA, no expression hubs (Bias~1 hub removed)"),
        ("ULTRA (Direct Overlap Masked)", r"ULTRA, direct \texttt{causes\_phenotype} masked (Bias~3 removed)"),
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
        ("Phrank-HPO (disease_raw)", r"Phrank, HPO disease-bridge (\texttt{disease\_raw})"),
        ("Phrank-HPO (direct_raw)", r"Phrank, HPO direct gene-union (\texttt{direct\_raw})"),
        ("Phrank-UPheno (Human+MGI Disease Raw)", r"Phrank, refitted UPheno H+MGI disease-max (Bias~2 fixed)"),
        ("Phrank-UPheno (Human+MGI Direct Raw)", r"Phrank, refitted UPheno H+MGI direct (Bias~2 fixed)"),
        ("Phrank-UPheno (MGI-Only Raw)", r"Phrank, refitted UPheno MGI-only (Biases~1--3 removed)"),
        ("PhenoRank (Static 2016 1000-Sim TB)", r"PhenoRank, static 2016 matrix, 1,000-sim.\ $p$-val.\ (TB)"),
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
        r"\caption{Step-by-step correction of architectural, training-status, phenotype-overlap, and profile-duplication biases across learned graph models (Panel A, including matched disease-bridge-free retrainings across all ten folds) and external or non-trained baselines (Panel B, including fold-specific refittings of Phrank and PhenoRank on our UPheno v2 and MGI annotations) over all ten folds (13,625 candidate genes). Each cell reports $\mathrm{H@10~(\%)}~/~\mathrm{MR}$. Step~0 (S0) uses raw scores on the full benchmark ($n=6{,}571$ pairs). Step~1 (S1) applies leave-one-disease-out per-gene reference calibration. Step~2 (S2) applies within-query status-stratum standardization (\texttt{Status-Z}), equalizing the mean and variance of seen ($3{,}779.2$) and unseen ($9{,}845.8$) candidate pools per query. Step~3 (S3) restricts evaluation to the zero-direct-overlap cohort ($J_{\mathrm{same}}=0$, $n=4{,}431$ pairs: $945$ seen pairs with zero shared training phenotypes plus $3{,}486$ unseen pairs). Step~4 (S4) additionally deduplicates identical phenotype sets ($P_d=P_{d'}$) in the calibration panel, excludes the 210 exact training-profile duplicates ($J_{\mathrm{any}}=1$), filters co-causal genes of multi-gene diseases, and weights unique disease profiles equally ($n=4{,}011$ unique diseases). The final column reports exact within-status calibrated mean ranks (Seen out of $3{,}779.2$; Unseen out of $9{,}845.8$).}",
        r"\label{tab:bias_stepwise}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method and architectural / annotation correction & S0: Raw & S1: +Per-gene Cal. & S2: +Status-Z & S3: +$J_{\mathrm{same}}{=}0$ & S4: +Dedup/Filt/EqD & Within-status MR \\",
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
    supp_lines = [
        r"\begin{table*}[tb]",
        r"\centering\scriptsize",
        r"\caption{Detailed seen-versus-unseen and overlap-stratified decomposition of the step-by-step bias corrections across learned models and baselines (ten folds, 13,625 candidates). Each cell reports $\mathrm{H@1~(\%)}~/~\mathrm{H@10~(\%)}~/~\mathrm{MR}$. Columns compare Step~1 per-gene calibration on seen ($n=3{,}085$) versus unseen ($n=3{,}486$) test pairs, Step~2 within-query status-stratum standardization (\texttt{Status-Z}) on seen versus unseen test pairs, Step~3 zero-direct-overlap evaluation ($J_{\mathrm{same}}=0$, $n=4{,}431$) under standard calibration (S3-Cal) versus status-standardized scoring (S3-StatusZ), and Step~4 profile-duplication and multi-gene correction applied in isolation to Step~1 calibrated scores across all non-duplicate test diseases (S4-on-S1, $n=5{,}878$ unique diseases).}",
        r"\label{tab:bias_baselines_full}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method variant & S1 Seen & S1 Unseen & S2 Seen & S2 Unseen & S3-Cal ($J_{\mathrm{same}}{=}0$) & S3-StatusZ & S4-on-S1 (Iso.) \\",
        r" & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR & H@1/H@10/MR \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\emph{Panel A: Learned knowledge-graph embedding and relational graph foundation models}} \\",
    ]

    for key, label in panel_a:
        m = d[key]
        supp_lines.append(
            f"{label} & {fmt_h1_h10_mr(m['S1_Cal_Seen'])} & {fmt_h1_h10_mr(m['S1_Cal_Unseen'])} & "
            f"{fmt_h1_h10_mr(m['S2_StatusZ_Seen'])} & {fmt_h1_h10_mr(m['S2_StatusZ_Unseen'])} & "
            f"{fmt_h1_h10_mr(m['S3_ZeroDirectOverlap_Cal'])} & {fmt_h1_h10_mr(m['S3_ZeroDirectOverlap_StatusZ'])} & "
            f"{fmt_h1_h10_mr(m['S4_Isolated_Dedup_Filt_EqD_on_S1'])} \\\\"
        )

    supp_lines.extend([
        r"\midrule",
        r"\multicolumn{8}{l}{\emph{Panel B: External ranking methods and non-trained semantic-similarity baselines}} \\",
    ])

    for key, label in panel_b:
        m = d[key]
        supp_lines.append(
            f"{label} & {fmt_h1_h10_mr(m['S1_Cal_Seen'])} & {fmt_h1_h10_mr(m['S1_Cal_Unseen'])} & "
            f"{fmt_h1_h10_mr(m['S2_StatusZ_Seen'])} & {fmt_h1_h10_mr(m['S2_StatusZ_Unseen'])} & "
            f"{fmt_h1_h10_mr(m['S3_ZeroDirectOverlap_Cal'])} & {fmt_h1_h10_mr(m['S3_ZeroDirectOverlap_StatusZ'])} & "
            f"{fmt_h1_h10_mr(m['S4_Isolated_Dedup_Filt_EqD_on_S1'])} \\\\"
        )

    supp_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
    ])
    OUT_SUPP.write_text("\n".join(supp_lines) + "\n")
    print("Generated both LaTeX tables.")


if __name__ == "__main__":
    main()
