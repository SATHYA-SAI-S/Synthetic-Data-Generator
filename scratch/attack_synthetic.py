"""
Adversarial audit: attempt to re-identify real patients from generated synthetic data.
Attacks:
  1. Exact-match re-identification on quasi-identifier subsets.
  2. Nearest-neighbour distance-based membership inference (D-MIA).
  3. Uniqueness analysis (how many synth rows are unique fingerprints).
"""
import zipfile
import numpy as np
import pandas as pd

REAL_ZIP = "data/diabetes+130-us+hospitals+for+years+1999-2008.zip"
SYNTH_FILES = {
    0.1: "scratch/sweep_results/synthetic_eps_0.1.csv",
    1.0: "scratch/sweep_results/synthetic_eps_1.0.csv",
    10.0: "scratch/sweep_results/synthetic_eps_10.0.csv",
}

with zipfile.ZipFile(REAL_ZIP) as z:
    with z.open("diabetic_data.csv") as f:
        real = pd.read_csv(f)

real = real.drop(columns=[c for c in ["encounter_id", "patient_nbr", "readmitted",
                                      "max_glu_serum", "A1Cresult"] if c in real.columns],
                errors="ignore")

QI_SETS = [
    ["race", "gender", "age"],
    ["race", "gender", "age", "admission_type_id"],
    ["race", "gender", "age", "time_in_hospital"],
    ["gender", "age", "num_medications"],
]

def factorize(df, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if df[c].dtype == object:
            out[c] = pd.factorize(df[c].astype(str))[0]
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return out.to_numpy(dtype=float)

for eps, path in SYNTH_FILES.items():
    synth = pd.read_csv(path)
    common = [c for c in real.columns if c in synth.columns]
    r, s = real[common], synth[common]
    print(f"===== eps={eps} | synth rows={len(s)} =====")

    # Attack 1: exact QI matches
    for qi in QI_SETS:
        qi = [c for c in qi if c in common]
        rq = set(map(tuple, r[qi].astype(str).values))
        sq = list(map(tuple, s[qi].astype(str).values))
        hits = sum(1 for t in sq if t in rq)
        print(f"  Exact QI match {qi}: {hits}/{len(sq)} synth rows "
              f"({100*hits/len(s):.2f}%) map to >=1 real record")

    # Attack 2: D-MIA nearest neighbour
    cols_num = factorize(r, common)
    synth_num = factorize(s, common)
    n = min(20000, len(cols_num))
    rng = np.random.default_rng(0)
    idx_tr = rng.choice(len(cols_num), n, replace=False)
    idx_ho = rng.choice(len(cols_num), n, replace=False)
    synth_num32 = synth_num.astype(np.float32)
    s_sq = (synth_num32 ** 2).sum(1)
    def nn1_dist(queries):
        # brute-force nearest neighbour via |q-s|^2 = |q|^2+|s|^2-2q.s (no sklearn)
        q = queries.astype(np.float32)
        d = np.empty(len(q), dtype=np.float32)
        for i in range(0, len(q), 1024):
            qb = q[i:i+1024]
            dists = (qb**2).sum(1)[:, None] + s_sq[None, :] - 2.0 * qb @ synth_num32.T
            d[i:i+len(qb)] = np.sqrt(np.maximum(dists.min(axis=1), 0))
        return float(d.mean())
    d_tr = nn1_dist(cols_num[idx_tr])
    d_ho = nn1_dist(cols_num[idx_ho])
    risk = (d_ho - d_tr) / max(d_ho, 1e-9)
    print(f"  NN dist train->synth={d_tr:.4f} holdout->synth={d_ho:.4f} risk={risk:+.3f}")

    # Attack 3: row uniqueness of synth
    uq = len(s.drop_duplicates()) / len(s)
    print(f"  Synth unique-row fraction: {uq:.4f}")

print("Done.")