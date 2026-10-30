"""Write every table of the final report from the results files into docs/tables/.

Each table is a LaTeX tabular fragment that docs/final.tex inputs, so no number in a
table is typed by hand. Tables whose inputs are missing are skipped with a message.

    python -m scripts.final_tables [OUTPUT_DIRECTORY]
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.predictor_study import targets
from scripts.shapley_attribution import is_primary, shapley

A = Path("results/setting-a-2026-10-18")
V5 = Path("results/setting-a-2026-09-26/research-v5")
T = Path("results/transfer-2026-10-20")
SWITCHES = ["idf", "saturation", "length"]
KEYS = [f"{i:03b}" for i in range(8)]
NAMES = {"msmarco_dev": "MS MARCO dev", "trec_dl_2019": "TREC-DL 19", "trec_dl_2020": "TREC-DL 20",
         "arguana": "ArguAna", "fiqa": "FiQA", "nfcorpus": "NFCorpus", "scidocs": "SCIDOCS", "scifact": "SciFact",
         "trec_covid": "TREC-COVID", "limit": "LIMIT", "miracl_hi": "MIRACL hi", "miracl_bn": "MIRACL bn",
         "miracl_te": "MIRACL te"}
ORDER = list(NAMES)
BASES = ["identity", "gaussian", "pca", "kmeans", "pq", "rq", "sae"]
BASIS_NAMES = {"identity": "identity", "gaussian": "Gaussian", "pca": "PCA", "kmeans": "$k$-means", "pq": "PQ",
               "rq": "RQ", "sae": "SAE"}
TASK_NAMES = {"niah_single_1": "NIAH single 1", "niah_single_2": "NIAH single 2", "niah_single_3": "NIAH single 3",
              "niah_multikey_1": "NIAH multikey 1", "niah_multikey_2": "NIAH multikey 2",
              "niah_multikey_3": "NIAH multikey 3", "niah_multivalue": "NIAH multivalue",
              "niah_multiquery": "NIAH multiquery", "vt": "variable tracking", "cwe": "common words",
              "fwe": "frequent words", "qa_1": "QA (SQuAD)", "qa_2": "QA (HotpotQA)"}


def f3(x):
    """Three decimals without the leading zero, as in the mid report: .123, $-.045$."""
    if x is None or np.isnan(x):
        return "--"
    x = 0.0 if round(x, 3) == 0 else x
    s = f"{x:.3f}"
    s = s.replace("0.", ".", 1) if abs(x) < 1 else s
    return f"${s}$" if s.startswith("-") else s


def f2(x):
    if x is None or np.isnan(x):
        return "--"
    x = 0.0 if round(x, 2) == 0 else x
    s = f"{x:.2f}"
    s = s.replace("0.", ".", 1) if abs(x) < 1 else s
    return f"${s}$" if s.startswith("-") else s


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def write(out, name, lines):
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", name)


def load_rows():
    return json.loads((A / "analysis" / "rows.json").read_text())


def primary(rows):
    return [r for r in rows if not r.get("control") and is_primary(r)]


def table_families(out, rows):
    count = lambda pred: sum(1 for r in rows if pred(r))
    default = lambda r: r["k1"] == 1.2 and r["b"] == 0.75 and r["seed"] == 0 and not r.get("control")
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Sweep & Coll. & Rows \\", r"\midrule"]
    families = [
        ("primary, 14 codes", lambda r: default(r) and r["budget"] == "q32_d128"),
        ("support budgets 16/64, 64/256", lambda r: default(r) and r["budget"] in ("q16_d64", "q64_d256")),
        ("native support", lambda r: default(r) and r["budget"] == "native"),
        (r"$k_1 \in \{.6, 1.2, 2.4\}$, $b \in \{0, .5, .75\}$",
         lambda r: not r.get("control") and r["seed"] == 0 and r["budget"] == "q32_d128" and (r["k1"], r["b"]) != (1.2, 0.75)),
        ("basis seeds 1 and 2", lambda r: not r.get("control") and r["seed"] != 0),
        (r"shuffled, uniform $df$ controls", lambda r: r.get("control") in ("shuffled_idf", "uniform_frequency")),
        ("shared SAE dictionary", lambda r: r.get("control") == "shared_dictionary"),
    ]
    for label, pred in families:
        members = [r for r in rows if pred(r)]
        lines.append(f"{label} & {len({r['collection'] for r in members})} & {thousands(len(members))} \\\\")
    lines += [r"\midrule", f"all rows & 13 & {thousands(len(rows))} \\\\", r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_families", lines)


def splade_quality():
    path = A / "splade" / "splade_v3_eight_configs.csv"
    result = {}
    for r in csv.DictReader(path.open()):
        result[r["collection"]] = {k: float(r[f"ndcg@10_idf{k[0]}_sat{k[1]}_len{k[2]}"]) for k in KEYS}
    return result


def table_quality(out, rows):
    baselines = defaultdict(dict)
    for b in json.loads((V5 / "report" / "baselines.json").read_text()):
        if b["method"] in ("bm25", "dense"):
            baselines[b["collection"]][b["method"]] = b["metrics"]["ndcg@10"]
    splade = splade_quality()
    p = primary(rows)
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Baselines} & \multicolumn{3}{c}{E5 sparse codes} & \multicolumn{2}{c}{SPLADE-v3} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-8}",
             r"Collection & BM25 & E5 & \texttt{000} & \texttt{111} & best & \texttt{000} & \texttt{111} \\", r"\midrule"]
    for c in ORDER:
        rs = [r for r in p if r["collection"] == c]
        off = np.mean([r["quality"]["000"] for r in rs])
        on = np.mean([r["quality"]["111"] for r in rs])
        best = max(r["quality"]["111"] for r in rs)
        s = splade.get(c, {})
        lines.append(f"{NAMES[c]} & {f3(baselines[c].get('bm25'))} & {f3(baselines[c].get('dense'))} & {f3(off)} & "
                     f"{f3(on)} & {f3(best)} & {f3(s.get('000'))} & {f3(s.get('111'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_quality", lines)


def shapley_rows():
    return list(csv.DictReader((A / "exploratory" / "shapley_rows.csv").open()))


def table_shapley(out, rows):
    """Mean Shapley value of each switch and their sum, in each setting's own metric."""
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Setting & Rows & IDF & Sat. & Len. & Sum \\", r"\midrule"]

    def add(label, n, values):
        v = np.asarray(values, float)
        lines.append(f"{label} & {n} & {f3(v[:, 0].mean())} & {f3(v[:, 1].mean())} & {f3(v[:, 2].mean())} & "
                     f"{f3(v.sum(1).mean())} \\\\")

    p = primary(rows)
    add("E5 codes, primary", len(p), [[shapley(r["quality"])[s] for s in SWITCHES] for r in p])
    for budget, label in (("q16_d64", "E5 codes, 16/64"), ("q64_d256", "E5 codes, 64/256"),
                          ("native", "E5 codes, native")):
        rs = [r for r in rows if not r.get("control") and r["seed"] == 0 and r["k1"] == 1.2 and r["b"] == 0.75
              and r["budget"] == budget]
        add(label, len(rs), [[shapley(r["quality"])[s] for s in SWITCHES] for r in rs])
    splade = splade_quality()
    sp = [c for c in ORDER if c in splade and not c.startswith("miracl")]
    add("SPLADE-v3", len(sp), [[shapley(splade[c])[s] for s in SWITCHES] for c in sp])
    b = T / "setting_b_study.json"
    if b.exists():
        study = json.loads(b.read_text())
        rs = [r for r in study["per_row"] if r["margin"] is not None]
        add("KV pages, Qwen", len(rs), [[shapley(r["quality"])[s] for s in SWITCHES] for r in rs])
    b = T / "setting_b_llama_study.json"
    if b.exists():
        study = json.loads(b.read_text())
        rs = [r for r in study["per_row"] if r["margin"] is not None]
        add("KV pages, Llama", len(rs), [[shapley(r["quality"])[s] for s in SWITCHES] for r in rs])
    c = T / "setting_c_study.json"
    if c.exists():
        study = json.loads(c.read_text())
        rs = [r for r in study["per_row"] if r["benchmark"] == "mmlu" and r["margin"] is not None]
        if rs:
            add("experts, MMLU", len(rs), [[shapley(r["quality"])[s] for s in SWITCHES] for r in rs])
        w = [r for r in study["per_row"] if r["benchmark"] == "wikitext"]
        if w:
            add("experts, WikiText", 1, [[shapley(w[0]["quality"])[s] for s in SWITCHES]])
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_shapley", lines)


def table_confirm(out):
    d = json.loads((A / "exploratory" / "confirmatory_tests.json").read_text())
    lines = [r"\begin{tabular}{@{}llr@{}}", r"\toprule", r" & Hypothesis & Result \\", r"\midrule",
             f"H1 & IDF + length share at default in $[.36, .42]$ & {f3(d['H1']['collection_statistics_share'])} \\\\",
             f"H2 & $\\phi_{{\\mathrm{{sat}}}}$ at $k_1 = 0.6$ exceeds $k_1 = 2.4$ & "
             f"{f3(d['H2']['mean_k1_0.6'])} $>$ {f3(d['H2']['mean_k1_2.4'])} \\\\",
             f"H3 & within-corpus $\\rho(\\mathrm{{cv}}_x, \\phi_{{\\mathrm{{sat}}}}) \\ge 0.3$ & "
             f"{f2(d['H3']['0.6']['rho'])}, {f2(d['H3']['2.4']['rho'])} \\\\",
             f"H4 & $\\mathrm{{cv}}_x$ model ranks $\\phi_{{\\mathrm{{sat}}}}$ at $\\rho \\ge 0.5$ & "
             f"{f2(d['H4']['0.6']['spearman'])}, {f2(d['H4']['2.4']['spearman'])} \\\\",
             f"H5 & $\\phi_{{\\mathrm{{len}}}} = 0$ exactly at $b = 0$ & max ${d['H5']['max_abs_length_shapley']:.1f}$ \\\\",
             f"H6 & $\\phi_{{\\mathrm{{len}}}}$ at $b = 0.75$ exceeds $b = 0.5$ & "
             f"{f3(d['H6']['mean_b_0.75'])} $>$ {f3(d['H6']['mean_b_0.5'])} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_confirm", lines)


def table_predictor(out):
    studies = {ref: json.loads((A / "exploratory" / f"predictor_study{suffix}.json").read_text())
               for ref, suffix in (("bm25", ""), ("dense", "_dense"))}
    labels = {"saturation": r"$\phi_{\mathrm{sat}}$", "length": r"$\phi_{\mathrm{len}}$",
              "idf": r"$\phi_{\mathrm{idf}}$", "collection_gain": "$G$"}
    tiers = [("pre-registered / ridge", "frequency curve", "bm25"), ("index / ridge", "index", "bm25"),
             ("pseudo-labels / ridge", "pseudo-labels", "bm25"), ("pseudo-labels / ridge", "pseudo-labels, E5", "dense")]
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"Target & Features & Family & Corpus & Params \\", r"\midrule"]
    for target, label in labels.items():
        for i, (key, name, ref) in enumerate(tiers):
            if ref == "dense" and target != "collection_gain":
                continue
            entry = studies[ref]["targets"][target][key]
            lines.append(f"{label if i == 0 else ''} & {name} & {f2(entry['family']['r2'])} & "
                         f"{f2(entry['corpus']['r2'])} & {f2(entry['other parameters']['r2'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_predictor", lines)


def table_bases(out, rows):
    p = primary(rows)
    diagnostics = json.loads((A / "exploratory" / "mechanism_diagnostics.json").read_text())
    cv = defaultdict(list)
    for d in diagnostics:
        if d["budget"] == "q32_d128" and d["seed"] == 0 and d.get("value_cv") is not None:
            cv[d["representation"]].append(d["value_cv"])
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Basis & Gran. & \texttt{111} & $\mathrm{cv}_x$ & $\phi_{\mathrm{sat}}$ & IDF + len. & $G > 0$ \\", r"\midrule"]
    for basis in BASES:
        for g in ("pooled", "token"):
            rs = [r for r in p if r["basis"] == basis and r["granularity"] == g]
            sh = np.array([[shapley(r["quality"])[s] for s in SWITCHES] for r in rs])
            gain = sh.sum(1)
            share = (sh[:, 0].sum() + sh[:, 2].sum()) / gain.sum() if gain.sum() > 0 else float("nan")
            positive = np.mean([targets(r["quality"])["collection_gain"] > 0 for r in rs])
            c = cv.get(f"{basis}__{g}")
            lines.append(f"{BASIS_NAMES[basis] if g == 'pooled' else ''} & {g} & {f3(np.mean([r['quality']['111'] for r in rs]))} & "
                         f"{np.mean(c):.2f} & {f3(sh[:, 1].mean())} & {f2(share)} & {f2(positive)} \\\\" if c else
                         f"{BASIS_NAMES[basis] if g == 'pooled' else ''} & {g} & -- & -- & -- & -- & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_bases", lines)


def table_attention(out):
    """Mean RULER accuracy per model and budget, and the share of exact attention mass on each configuration's pages."""
    studies = [("Qwen", T / "setting_b_study.json", T / "setting-b" / "attention_mass"),
               ("Llama", T / "setting_b_llama_study.json", T / "setting-b-llama" / "attention_mass")]
    studies = [(m, json.loads(s.read_text()), d) for m, s, d in studies if s.exists()]
    if not studies:
        print("skip tab_attention: no study")
        return
    modes = ["full", "quest", "h2o"] + KEYS
    head = ["Full", "Quest", "H2O"] + [rf"\texttt{{{k}}}" for k in KEYS]
    lines = [r"\begin{tabular}{@{}lll" + "r" * len(modes) + r"@{}}", r"\toprule",
             "Model & & Budget & " + " & ".join(head) + r" \\"]
    for model, study, mass_dir in studies:
        lines.append(r"\midrule")
        budgets = sorted((k for k in study["mean_accuracy"] if k.startswith("budget_")), key=lambda k: int(k.split("_")[1]))
        for i, b in enumerate(budgets):
            m = study["mean_accuracy"][b]
            lines.append(f"{model if i == 0 else ''} & {'Accuracy' if i == 0 else ''} & {b.split('_')[1]} & "
                         + " & ".join(f3(m[k]) for k in modes) + r" \\")
        mass = [json.loads(f.read_text()) for f in sorted(mass_dir.glob("*.json"))]
        mass = [r for r in mass if r["pseudo_quality"] is not None]
        for i, b in enumerate(budgets):
            rs = [r for r in mass if r["budget"] == int(b.split("_")[1])]
            cells = ["1", f3(np.mean([r["pseudo_quality"]["000"] for r in rs])), "--"]
            cells += [f3(np.mean([r["pseudo_quality"][k] for r in rs])) for k in KEYS]
            lines.append(f" & {'Mass' if i == 0 else ''} & {b.split('_')[1]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_attention", lines)


def table_routing(out):
    """Mean MMLU accuracy over the 57 subjects and WikiText perplexity, merging configurations that route identically."""
    path = T / "setting_c_study.json"
    if not path.exists():
        print("skip tab_routing: no", path)
        return
    study = json.loads(path.read_text())
    m = study["mean_accuracy_mmlu"]
    w = study.get("wikitext")
    ppl = ({k: np.exp(-w["baselines"][k]) for k in ("native", "frequency_bias")} | w["perplexity"]) if w else {}
    groups = [(r"native, \texttt{000}--\texttt{011}", ["native", "000", "001", "010", "011"]),
              ("frequency bias", ["frequency_bias"]),
              (r"\texttt{100}, \texttt{101}", ["100", "101"]), (r"\texttt{110}", ["110"]), (r"\texttt{111}", ["111"])]
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Routing & MMLU acc. & WikiText ppl. \\", r"\midrule"]
    for label, keys in groups:
        # The native path runs the model's own forward, so its WikiText loss differs from 000 at the sixth digit.
        if len({round(m[k], 12) for k in keys}) != 1 or (ppl and max(ppl[k] for k in keys) - min(ppl[k] for k in keys) > 1e-4):
            raise ValueError(f"{keys} do not route identically")
        k = keys[0]
        lines.append(f"{label} & {f3(m[k])} & " + (f"{ppl[k]:.2f}" if ppl else "--") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_routing", lines)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/tables")
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    table_families(out, rows)
    table_quality(out, rows)
    table_shapley(out, rows)
    table_confirm(out)
    table_predictor(out)
    table_bases(out, rows)
    table_attention(out)
    table_routing(out)


if __name__ == "__main__":
    main()
