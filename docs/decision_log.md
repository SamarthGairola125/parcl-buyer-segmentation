# Decision Log

| # | Decision | Rationale |
|---|---|---|
| D1 | Age measured at 2025-12-31 (last transaction month) | Reproducible; does not drift with run date |
| D2 | DOB parsed by unifying `/` to `-`, then `%m-%d-%Y` | All three source formats are month-first; 0 parse failures |
| D3 | First and last names dropped at load | Personal data; not predictive |
| D4 | Modelling unit = client (2,000 rows) | The brief segments buyers, not units |
| D5 | Region excluded from clustering; used for profiling and location quotients | 57 sparse levels nested in country would dominate distances |
| D6 | Binary fields label-encoded 0/1; referral channel one-hot (Website baseline) | Label-encoding a nominal field imposes a false order |
| D7 | Numerics standardised; binaries left 0/1; MinMaxScaler run as sensitivity check | One category difference ≈ one standard deviation |
| D8 | Behavioural features built from `properties.csv` | The brief's "multiple units" and "large investments" are untestable without them |
| D9 | Brief segments C1–C4 treated as hypotheses, not targets | Independence audit showed near-zero association between attributes |
| D10 | Three feature sets compared (brief / behaviour / primary) | Separates what the brief fields contain from what behaviour adds |
| D11 | k = 4 for the primary model | Silhouette flat for k = 2–4; dendrogram and stability support 4; maps to the brief's four segments |
| D12 | Validation by permutation null, gap statistic and bootstrap Jaccard | Elbow and Silhouette alone cannot detect "no structure" |
| D13 | Segment names assigned by explicit profile rules | Names must not depend on arbitrary K-Means label order |
| D14 | Hypotheses tested with Benjamini–Hochberg correction; n always reported | Multiple testing across 13 hypotheses |
| D15 | Dashboard reads saved segments; no retraining in the app | Fast, light, consistent with the paper |
| D16 | Random seed 42 throughout | Reproducibility |
| D17 | 3D Buyer Galaxy uses PCA (not UMAP or t-SNE), stored as a separate versioned file `embedding_3d_v1.csv` | PCA is deterministic and its axes can be explained; the model and published 2D coordinates stay unchanged (tested) |
| D18 | Every 3D view has a 2D toggle; no auto-rotation | Exact reading and reduced-motion users |
| D19 | Rates show k of n and a 95% Wilson interval; n < 100 flagged as a small sample | Honest uncertainty; Wilson behaves well for small n |
| D20 | Headlines generated from data by templates, with unit tests | Titles stay correct under any filter |
| D21 | Two tags: "Observed in data" and "Model output" | Separates counted facts from clustering-dependent results |
| D22 | Previous dashboard kept as `app/streamlit_app_v1.py` | Non-destructive upgrade; easy rollback |

## Hypothesis register

Results are in `results/hypothesis_tests.csv`. H1–H5 were glanced at during the data audit and are reported as exploratory.

| ID | Hypothesis | Result (BH-adjusted) |
|---|---|---|
| H1 | Investment buyers are younger | Not supported |
| H2 | Investment buyers borrow more | Not supported |
| H3 | International buyers invest more | Not supported |
| H4 | Companies buy more units (C3) | Not supported |
| H5 | Company representatives differ in age | **Supported** (47.6 vs 56.0) |
| H6 | Satisfaction differs by referral channel | Not supported |
| H7 | Higher spenders more satisfied (C4) | Not supported |
| H8a/b | Investors / companies hold more offices | Not supported |
| H9 | Loan applicants buy cheaper units | Not supported |
| H10 | Cross-year repeat buyers spend more | **Supported** |
| H11 | Data has more structure than a shuffled null | Brief fields: no. Primary: **yes** |
| H12 | At least one segment is stable (Jaccard ≥ 0.75) | **Supported** (3 of 4) |
| H13 | Behaviour segments separate non-input actionable traits | Not supported (only age band, marginally) |
| H14 | International buyers spend differently | **Supported** (3% less; small) |
| H15 | Loan applicants are younger (C2) | Not supported |
