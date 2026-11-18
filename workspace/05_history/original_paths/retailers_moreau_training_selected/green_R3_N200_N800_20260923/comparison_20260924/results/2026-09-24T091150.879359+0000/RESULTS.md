# Fixed-policy R3 comparison

Status: **failed**.
No new training, CV, or test-based selection. Green training counts 40/160/640; SCS 50/200/800.
Fresh strict green N200/N800 (180 policies) and SCS N50/N200/N800 (600 policies).
Green N50 (90 policies), SAA-kNN, and SRO-FA are cached historical references.
OOS: fixed seed 9999, 10,000 float64 samples. Previously inspected test: descriptive comparison.
IS: mean of per-center maxima on the returned checkpoint finite pool, not an adversarial supremum.
Green membership gate remains 1e-9. SCS saved coordinates are unchanged and not box-certified.

Host: gold-srv2.iem.technion.ac.il; affinity CPUs: 72; spawn workers: 32; one numerical/solver thread each.
Preflight: /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/preflight_v2/summary.json; SHA256: f7b52ffc49d77e3edc9c242d6a1b4ee0f0588031e9f15292a5ad80f1a5c25e84.
Manifest: frozen/manifest.json; SHA256: adb1aceb87f01191db952e8054825895094d8dc54c890c5761cd8bd1c973d5fa.
Final source hashes verified: False.
Current pipeline elapsed before reporting: 92.822348 s / 1.547039 min / 0.025784 h (92.822348 s).
Raw current times: policies/*/result.json; journals: policies/*/{oos,is}_solves.jsonl.
Per-model backend wall times exclude final result serialization; overall sweep includes dispatch and worker I/O.
Training references and AP/MP iteration timings remain in the checksummed frozen original summaries.

## Current evaluation wall times (seconds)

| Method | N | Radius | Status | Scope | n | Mean | Median | Min | Max |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|

## Persisted historical training / reference timings

These are old-run times, never current cache-loading times. Historical hardware/concurrency is only
known where recorded in the frozen source; otherwise unavailable. Missing times require a rerun.

| Method / scope | N | Radius / LR | n | Mean s | Median s | Min s | Max s |
|---|---:|---|---:|---:|---:|---:|---:|
| Green nominal training | 200 | LR=0.005 | 10 | 145.9561 | 148.5459 | 130.2067 | 150.8644 |
| Green nominal train evaluation | 200 | LR=0.005 | 10 | 0.0177 | 0.0175 | 0.0073 | 0.0241 |
| Green nominal training | 200 | LR=0.01 | 10 | 146.9329 | 149.6870 | 120.6584 | 150.6502 |
| Green nominal train evaluation | 200 | LR=0.01 | 10 | 0.0134 | 0.0118 | 0.0087 | 0.0233 |
| Green nominal training | 200 | LR=0.05 | 10 | 149.8254 | 149.8104 | 149.0422 | 150.4982 |
| Green nominal train evaluation | 200 | LR=0.05 | 10 | 0.0111 | 0.0105 | 0.0092 | 0.0138 |
| Green CCG total (excludes shared nominal) | 200 | 0.0 | 10 | 4.1014 | 4.0585 | 3.9375 | 4.3861 |
| Green CCG total (excludes shared nominal) | 200 | 0.001 | 10 | 4.2957 | 4.2710 | 4.1132 | 4.5220 |
| Green CCG total (excludes shared nominal) | 200 | 0.003 | 10 | 4.3409 | 4.3626 | 4.1507 | 4.5066 |
| Green CCG total (excludes shared nominal) | 200 | 0.01 | 10 | 4.3867 | 4.3936 | 4.2289 | 4.5421 |
| Green CCG total (excludes shared nominal) | 200 | 0.03 | 10 | 4.4514 | 4.4511 | 4.1495 | 4.6857 |
| Green CCG total (excludes shared nominal) | 200 | 0.2 | 10 | 292.7897 | 285.9686 | 143.3843 | 450.2265 |
| Green CCG total (excludes shared nominal) | 200 | 0.4 | 10 | 434.9543 | 450.4736 | 242.6846 | 581.9816 |
| Green CCG total (excludes shared nominal) | 200 | 0.7 | 10 | 600.5187 | 577.8339 | 423.9759 | 787.1546 |
| Green CCG total (excludes shared nominal) | 200 | 1.5 | 10 | 456.5732 | 450.9349 | 261.4758 | 647.9089 |
| Green initialization | 200 | all | 10 | 0.1129 | 0.1030 | 0.0789 | 0.1680 |
| Green lr_validation | 200 | all | 10 | 0.0522 | 0.0561 | 0.0371 | 0.0654 |
| Green radius_validation | 200 | all | 10 | 0.2050 | 0.2079 | 0.1610 | 0.2319 |
| Green nominal training | 800 | LR=0.005 | 10 | 172.0684 | 173.0375 | 159.0950 | 175.6961 |
| Green nominal train evaluation | 800 | LR=0.005 | 10 | 0.0108 | 0.0104 | 0.0089 | 0.0134 |
| Green nominal training | 800 | LR=0.01 | 10 | 172.4072 | 172.9799 | 167.7586 | 174.6118 |
| Green nominal train evaluation | 800 | LR=0.01 | 10 | 0.0121 | 0.0123 | 0.0088 | 0.0161 |
| Green nominal training | 800 | LR=0.05 | 10 | 160.3951 | 170.1300 | 78.2683 | 174.1930 |
| Green nominal train evaluation | 800 | LR=0.05 | 10 | 0.0102 | 0.0098 | 0.0078 | 0.0140 |
| Green CCG total (excludes shared nominal) | 800 | 0.0 | 10 | 15.9403 | 15.8439 | 15.5961 | 16.6094 |
| Green CCG total (excludes shared nominal) | 800 | 0.001 | 10 | 16.7674 | 16.8198 | 16.2809 | 17.0448 |
| Green CCG total (excludes shared nominal) | 800 | 0.003 | 10 | 16.8267 | 16.8345 | 16.4814 | 17.1370 |
| Green CCG total (excludes shared nominal) | 800 | 0.01 | 10 | 17.2715 | 17.1865 | 16.3486 | 18.3432 |
| Green CCG total (excludes shared nominal) | 800 | 0.03 | 10 | 18.2539 | 18.2060 | 17.3857 | 19.0915 |
| Green CCG total (excludes shared nominal) | 800 | 0.2 | 10 | 281.6136 | 262.9244 | 234.1574 | 484.0145 |
| Green CCG total (excludes shared nominal) | 800 | 0.4 | 10 | 528.9813 | 518.4895 | 344.7024 | 775.8673 |
| Green CCG total (excludes shared nominal) | 800 | 0.7 | 10 | 670.4103 | 719.4147 | 454.5906 | 801.3468 |
| Green CCG total (excludes shared nominal) | 800 | 1.5 | 10 | 662.3939 | 680.6623 | 517.2090 | 769.8383 |
| Green initialization | 800 | all | 10 | 0.2639 | 0.2651 | 0.2122 | 0.3397 |
| Green lr_validation | 800 | all | 10 | 0.0641 | 0.0678 | 0.0442 | 0.0778 |
| Green radius_validation | 800 | all | 10 | 0.2446 | 0.2321 | 0.2219 | 0.2881 |
| SCS nominal/full CCG (not certified) | 50 | all | 0 | unavailable | unavailable | unavailable | unavailable |
| SRO-FA historical partition | 50 | 0.001 | 10 | 0.0015 | 0.0016 | 0.0011 | 0.0018 |
| SRO-FA historical solver | 50 | 0.001 | 10 | 0.0028 | 0.0027 | 0.0020 | 0.0044 |
| SRO-FA historical OOS | 50 | 0.001 | 10 | 0.0300 | 0.0283 | 0.0244 | 0.0479 |
| SRO-FA historical total | 50 | 0.001 | 10 | 0.0745 | 0.0628 | 0.0548 | 0.1678 |
| SRO-FA historical partition | 50 | 0.0015 | 10 | 0.0016 | 0.0018 | 0.0011 | 0.0022 |
| SRO-FA historical solver | 50 | 0.0015 | 10 | 0.0027 | 0.0028 | 0.0017 | 0.0038 |
| SRO-FA historical OOS | 50 | 0.0015 | 10 | 0.0314 | 0.0316 | 0.0256 | 0.0371 |
| SRO-FA historical total | 50 | 0.0015 | 10 | 0.0680 | 0.0695 | 0.0531 | 0.0752 |
| SRO-FA historical partition | 50 | 0.002 | 10 | 0.0017 | 0.0017 | 0.0011 | 0.0023 |
| SRO-FA historical solver | 50 | 0.002 | 10 | 0.0030 | 0.0028 | 0.0022 | 0.0052 |
| SRO-FA historical OOS | 50 | 0.002 | 10 | 0.0294 | 0.0283 | 0.0241 | 0.0349 |
| SRO-FA historical total | 50 | 0.002 | 10 | 0.0643 | 0.0655 | 0.0512 | 0.0723 |
| SRO-FA historical partition | 50 | 0.003 | 10 | 0.0017 | 0.0018 | 0.0012 | 0.0021 |
| SRO-FA historical solver | 50 | 0.003 | 10 | 0.0032 | 0.0028 | 0.0019 | 0.0081 |
| SRO-FA historical OOS | 50 | 0.003 | 10 | 0.0290 | 0.0293 | 0.0257 | 0.0328 |
| SRO-FA historical total | 50 | 0.003 | 10 | 0.0767 | 0.0692 | 0.0643 | 0.1216 |
| SRO-FA historical partition | 50 | 0.005 | 10 | 0.0018 | 0.0019 | 0.0011 | 0.0022 |
| SRO-FA historical solver | 50 | 0.005 | 10 | 0.0027 | 0.0027 | 0.0021 | 0.0033 |
| SRO-FA historical OOS | 50 | 0.005 | 10 | 0.0297 | 0.0275 | 0.0232 | 0.0419 |
| SRO-FA historical total | 50 | 0.005 | 10 | 0.0649 | 0.0635 | 0.0507 | 0.0792 |
| SRO-FA historical partition | 50 | 0.01 | 10 | 0.0017 | 0.0017 | 0.0011 | 0.0020 |
| SRO-FA historical solver | 50 | 0.01 | 10 | 0.0034 | 0.0025 | 0.0019 | 0.0113 |
| SRO-FA historical OOS | 50 | 0.01 | 10 | 0.0295 | 0.0296 | 0.0260 | 0.0326 |
| SRO-FA historical total | 50 | 0.01 | 10 | 0.0721 | 0.0696 | 0.0570 | 0.0907 |
| SRO-FA historical partition | 50 | 0.02 | 10 | 0.0027 | 0.0018 | 0.0014 | 0.0108 |
| SRO-FA historical solver | 50 | 0.02 | 10 | 0.0030 | 0.0031 | 0.0021 | 0.0038 |
| SRO-FA historical OOS | 50 | 0.02 | 10 | 0.0318 | 0.0305 | 0.0251 | 0.0406 |
| SRO-FA historical total | 50 | 0.02 | 10 | 0.0719 | 0.0692 | 0.0568 | 0.0842 |
| SRO-FA historical partition | 50 | 0.05 | 10 | 0.0017 | 0.0018 | 0.0011 | 0.0030 |
| SRO-FA historical solver | 50 | 0.05 | 10 | 0.0035 | 0.0036 | 0.0019 | 0.0051 |
| SRO-FA historical OOS | 50 | 0.05 | 10 | 0.0298 | 0.0290 | 0.0272 | 0.0362 |
| SRO-FA historical total | 50 | 0.05 | 10 | 0.0729 | 0.0717 | 0.0541 | 0.1160 |
| SRO-FA historical partition | 50 | 0.1 | 10 | 0.0019 | 0.0018 | 0.0011 | 0.0036 |
| SRO-FA historical solver | 50 | 0.1 | 10 | 0.0043 | 0.0038 | 0.0028 | 0.0057 |
| SRO-FA historical OOS | 50 | 0.1 | 10 | 0.0298 | 0.0301 | 0.0222 | 0.0357 |
| SRO-FA historical total | 50 | 0.1 | 10 | 0.0769 | 0.0698 | 0.0600 | 0.1080 |
| SRO-FA historical partition | 50 | 0.15 | 10 | 0.0017 | 0.0017 | 0.0012 | 0.0030 |
| SRO-FA historical solver | 50 | 0.15 | 10 | 0.0055 | 0.0055 | 0.0034 | 0.0085 |
| SRO-FA historical OOS | 50 | 0.15 | 10 | 0.0286 | 0.0290 | 0.0242 | 0.0323 |
| SRO-FA historical total | 50 | 0.15 | 10 | 0.0772 | 0.0767 | 0.0658 | 0.0907 |
| SRO-FA historical partition | 50 | 0.2 | 10 | 0.0017 | 0.0018 | 0.0013 | 0.0020 |
| SRO-FA historical solver | 50 | 0.2 | 10 | 0.0067 | 0.0056 | 0.0046 | 0.0161 |
| SRO-FA historical OOS | 50 | 0.2 | 10 | 0.0285 | 0.0282 | 0.0251 | 0.0339 |
| SRO-FA historical total | 50 | 0.2 | 10 | 0.0863 | 0.0860 | 0.0753 | 0.0999 |
| SRO-FA historical partition | 50 | 0.3 | 10 | 0.0014 | 0.0013 | 0.0011 | 0.0019 |
| SRO-FA historical solver | 50 | 0.3 | 10 | 0.0072 | 0.0071 | 0.0041 | 0.0106 |
| SRO-FA historical OOS | 50 | 0.3 | 10 | 0.0280 | 0.0282 | 0.0246 | 0.0318 |
| SRO-FA historical total | 50 | 0.3 | 10 | 0.0887 | 0.0864 | 0.0655 | 0.1074 |
| SRO-FA historical partition | 50 | 0.4 | 10 | 0.0016 | 0.0015 | 0.0011 | 0.0025 |
| SRO-FA historical solver | 50 | 0.4 | 10 | 0.0097 | 0.0095 | 0.0072 | 0.0129 |
| SRO-FA historical OOS | 50 | 0.4 | 10 | 0.0301 | 0.0301 | 0.0239 | 0.0385 |
| SRO-FA historical total | 50 | 0.4 | 10 | 0.1020 | 0.0976 | 0.0890 | 0.1197 |
| SRO-FA historical partition | 50 | 0.45 | 10 | 0.0017 | 0.0017 | 0.0012 | 0.0020 |
| SRO-FA historical solver | 50 | 0.45 | 10 | 0.0102 | 0.0105 | 0.0075 | 0.0123 |
| SRO-FA historical OOS | 50 | 0.45 | 10 | 0.0291 | 0.0289 | 0.0238 | 0.0350 |
| SRO-FA historical total | 50 | 0.45 | 10 | 0.1104 | 0.1098 | 0.0851 | 0.1368 |
| SRO-FA historical partition | 50 | 0.5 | 10 | 0.0018 | 0.0018 | 0.0011 | 0.0024 |
| SRO-FA historical solver | 50 | 0.5 | 10 | 0.0142 | 0.0126 | 0.0097 | 0.0313 |
| SRO-FA historical OOS | 50 | 0.5 | 10 | 0.0325 | 0.0316 | 0.0278 | 0.0414 |
| SRO-FA historical total | 50 | 0.5 | 10 | 0.1248 | 0.1184 | 0.0988 | 0.1801 |
| SRO-FA historical partition | 50 | 0.55 | 10 | 0.0016 | 0.0016 | 0.0011 | 0.0019 |
| SRO-FA historical solver | 50 | 0.55 | 10 | 0.0147 | 0.0153 | 0.0103 | 0.0206 |
| SRO-FA historical OOS | 50 | 0.55 | 10 | 0.0276 | 0.0273 | 0.0231 | 0.0330 |
| SRO-FA historical total | 50 | 0.55 | 10 | 0.1203 | 0.1182 | 0.0923 | 0.1514 |
| SRO-FA historical partition | 50 | 0.6 | 10 | 0.0017 | 0.0017 | 0.0016 | 0.0020 |
| SRO-FA historical solver | 50 | 0.6 | 10 | 0.0161 | 0.0167 | 0.0116 | 0.0207 |
| SRO-FA historical OOS | 50 | 0.6 | 10 | 0.0312 | 0.0289 | 0.0251 | 0.0575 |
| SRO-FA historical total | 50 | 0.6 | 10 | 0.1267 | 0.1265 | 0.0991 | 0.1565 |
| SRO-FA historical partition | 50 | 0.65 | 10 | 0.0018 | 0.0016 | 0.0011 | 0.0026 |
| SRO-FA historical solver | 50 | 0.65 | 10 | 0.0172 | 0.0170 | 0.0125 | 0.0214 |
| SRO-FA historical OOS | 50 | 0.65 | 10 | 0.0275 | 0.0283 | 0.0212 | 0.0318 |
| SRO-FA historical total | 50 | 0.65 | 10 | 0.1423 | 0.1352 | 0.1134 | 0.2104 |
| SRO-FA historical partition | 50 | 0.7 | 10 | 0.0016 | 0.0016 | 0.0013 | 0.0021 |
| SRO-FA historical solver | 50 | 0.7 | 10 | 0.0204 | 0.0182 | 0.0158 | 0.0283 |
| SRO-FA historical OOS | 50 | 0.7 | 10 | 0.0338 | 0.0292 | 0.0235 | 0.0577 |
| SRO-FA historical total | 50 | 0.7 | 10 | 0.1446 | 0.1433 | 0.1179 | 0.1708 |
| SRO-FA historical partition | 50 | 0.75 | 10 | 0.0018 | 0.0019 | 0.0013 | 0.0025 |
| SRO-FA historical solver | 50 | 0.75 | 10 | 0.0208 | 0.0204 | 0.0160 | 0.0269 |
| SRO-FA historical OOS | 50 | 0.75 | 10 | 0.0318 | 0.0290 | 0.0255 | 0.0389 |
| SRO-FA historical total | 50 | 0.75 | 10 | 0.1656 | 0.1623 | 0.1288 | 0.2130 |
| SRO-FA historical partition | 50 | 0.8 | 10 | 0.0018 | 0.0018 | 0.0016 | 0.0021 |
| SRO-FA historical solver | 50 | 0.8 | 10 | 0.0249 | 0.0235 | 0.0171 | 0.0387 |
| SRO-FA historical OOS | 50 | 0.8 | 10 | 0.0280 | 0.0282 | 0.0234 | 0.0310 |
| SRO-FA historical total | 50 | 0.8 | 10 | 0.1631 | 0.1674 | 0.1167 | 0.1899 |
| SRO-FA historical partition | 50 | 1.0 | 10 | 0.0018 | 0.0019 | 0.0015 | 0.0020 |
| SRO-FA historical solver | 50 | 1.0 | 10 | 0.0348 | 0.0337 | 0.0252 | 0.0453 |
| SRO-FA historical OOS | 50 | 1.0 | 10 | 0.0293 | 0.0304 | 0.0232 | 0.0326 |
| SRO-FA historical total | 50 | 1.0 | 10 | 0.1900 | 0.1906 | 0.1577 | 0.2206 |
| SRO-FA historical partition | 50 | 1.5 | 10 | 0.0017 | 0.0018 | 0.0012 | 0.0023 |
| SRO-FA historical solver | 50 | 1.5 | 10 | 0.0585 | 0.0590 | 0.0474 | 0.0677 |
| SRO-FA historical OOS | 50 | 1.5 | 10 | 0.0281 | 0.0287 | 0.0225 | 0.0363 |
| SRO-FA historical total | 50 | 1.5 | 10 | 0.2758 | 0.2780 | 0.2423 | 0.3088 |
| SCS nominal/full CCG (not certified) | 200 | all | 0 | unavailable | unavailable | unavailable | unavailable |
| SRO-FA historical partition | 200 | 0.001 | 10 | 0.0075 | 0.0073 | 0.0056 | 0.0114 |
| SRO-FA historical solver | 200 | 0.001 | 10 | 0.0122 | 0.0121 | 0.0114 | 0.0132 |
| SRO-FA historical OOS | 200 | 0.001 | 10 | 0.1095 | 0.1070 | 0.1015 | 0.1211 |
| SRO-FA historical total | 200 | 0.001 | 10 | 0.2319 | 0.2270 | 0.2035 | 0.2814 |
| SRO-FA historical partition | 200 | 0.005 | 10 | 0.0084 | 0.0081 | 0.0074 | 0.0095 |
| SRO-FA historical solver | 200 | 0.005 | 10 | 0.0130 | 0.0129 | 0.0113 | 0.0149 |
| SRO-FA historical OOS | 200 | 0.005 | 10 | 0.1132 | 0.1096 | 0.1015 | 0.1577 |
| SRO-FA historical total | 200 | 0.005 | 10 | 0.2284 | 0.2211 | 0.2024 | 0.2771 |
| SRO-FA historical partition | 200 | 0.01 | 10 | 0.0087 | 0.0081 | 0.0073 | 0.0122 |
| SRO-FA historical solver | 200 | 0.01 | 10 | 0.0137 | 0.0139 | 0.0112 | 0.0166 |
| SRO-FA historical OOS | 200 | 0.01 | 10 | 0.1175 | 0.1151 | 0.1020 | 0.1413 |
| SRO-FA historical total | 200 | 0.01 | 10 | 0.2340 | 0.2365 | 0.2048 | 0.2658 |
| SRO-FA historical partition | 200 | 0.02 | 10 | 0.0083 | 0.0087 | 0.0061 | 0.0098 |
| SRO-FA historical solver | 200 | 0.02 | 10 | 0.0167 | 0.0168 | 0.0143 | 0.0182 |
| SRO-FA historical OOS | 200 | 0.02 | 10 | 0.1131 | 0.1092 | 0.1010 | 0.1439 |
| SRO-FA historical total | 200 | 0.02 | 10 | 0.2441 | 0.2400 | 0.2159 | 0.3013 |
| SRO-FA historical partition | 200 | 0.05 | 10 | 0.0081 | 0.0084 | 0.0060 | 0.0097 |
| SRO-FA historical solver | 200 | 0.05 | 10 | 0.0206 | 0.0202 | 0.0174 | 0.0239 |
| SRO-FA historical OOS | 200 | 0.05 | 10 | 0.1119 | 0.1128 | 0.0982 | 0.1411 |
| SRO-FA historical total | 200 | 0.05 | 10 | 0.2695 | 0.2658 | 0.2447 | 0.2931 |
| SRO-FA historical partition | 200 | 0.1 | 10 | 0.0084 | 0.0083 | 0.0057 | 0.0110 |
| SRO-FA historical solver | 200 | 0.1 | 10 | 0.0357 | 0.0357 | 0.0295 | 0.0426 |
| SRO-FA historical OOS | 200 | 0.1 | 10 | 0.1079 | 0.1058 | 0.0979 | 0.1202 |
| SRO-FA historical total | 200 | 0.1 | 10 | 0.2996 | 0.2989 | 0.2754 | 0.3363 |
| SRO-FA historical partition | 200 | 0.15 | 10 | 0.0078 | 0.0079 | 0.0054 | 0.0092 |
| SRO-FA historical solver | 200 | 0.15 | 10 | 0.0450 | 0.0448 | 0.0405 | 0.0503 |
| SRO-FA historical OOS | 200 | 0.15 | 10 | 0.1073 | 0.1049 | 0.0968 | 0.1268 |
| SRO-FA historical total | 200 | 0.15 | 10 | 0.3291 | 0.3260 | 0.2930 | 0.3657 |
| SRO-FA historical partition | 200 | 0.2 | 10 | 0.0084 | 0.0085 | 0.0062 | 0.0128 |
| SRO-FA historical solver | 200 | 0.2 | 10 | 0.0599 | 0.0589 | 0.0437 | 0.0730 |
| SRO-FA historical OOS | 200 | 0.2 | 10 | 0.1102 | 0.1093 | 0.1018 | 0.1238 |
| SRO-FA historical total | 200 | 0.2 | 10 | 0.3799 | 0.3827 | 0.3328 | 0.4266 |
| SRO-FA historical partition | 200 | 0.3 | 10 | 0.0083 | 0.0084 | 0.0062 | 0.0101 |
| SRO-FA historical solver | 200 | 0.3 | 10 | 0.0785 | 0.0758 | 0.0718 | 0.0921 |
| SRO-FA historical OOS | 200 | 0.3 | 10 | 0.1080 | 0.1069 | 0.0990 | 0.1280 |
| SRO-FA historical total | 200 | 0.3 | 10 | 0.4450 | 0.4389 | 0.4126 | 0.4866 |
| SRO-FA historical partition | 200 | 0.4 | 10 | 0.0080 | 0.0078 | 0.0066 | 0.0109 |
| SRO-FA historical solver | 200 | 0.4 | 10 | 0.1283 | 0.1335 | 0.0982 | 0.1543 |
| SRO-FA historical OOS | 200 | 0.4 | 10 | 0.1120 | 0.1114 | 0.0906 | 0.1398 |
| SRO-FA historical total | 200 | 0.4 | 10 | 0.5809 | 0.5781 | 0.5187 | 0.6426 |
| SRO-FA historical partition | 200 | 0.45 | 10 | 0.0073 | 0.0072 | 0.0055 | 0.0093 |
| SRO-FA historical solver | 200 | 0.45 | 10 | 0.1717 | 0.1713 | 0.1422 | 0.1897 |
| SRO-FA historical OOS | 200 | 0.45 | 10 | 0.1046 | 0.1017 | 0.0935 | 0.1183 |
| SRO-FA historical total | 200 | 0.45 | 10 | 0.6519 | 0.6421 | 0.5904 | 0.7278 |
| SRO-FA historical partition | 200 | 0.5 | 10 | 0.0085 | 0.0081 | 0.0053 | 0.0133 |
| SRO-FA historical solver | 200 | 0.5 | 10 | 0.2145 | 0.2153 | 0.1751 | 0.2613 |
| SRO-FA historical OOS | 200 | 0.5 | 10 | 0.1092 | 0.1105 | 0.0890 | 0.1191 |
| SRO-FA historical total | 200 | 0.5 | 10 | 0.7546 | 0.7388 | 0.6841 | 0.8480 |
| SRO-FA historical partition | 200 | 0.6 | 10 | 0.0084 | 0.0082 | 0.0053 | 0.0140 |
| SRO-FA historical solver | 200 | 0.6 | 10 | 0.2648 | 0.2678 | 0.2223 | 0.3039 |
| SRO-FA historical OOS | 200 | 0.6 | 10 | 0.1023 | 0.1043 | 0.0853 | 0.1164 |
| SRO-FA historical total | 200 | 0.6 | 10 | 0.8406 | 0.8539 | 0.7227 | 0.9301 |
| SRO-FA historical partition | 200 | 0.75 | 10 | 0.0082 | 0.0084 | 0.0057 | 0.0100 |
| SRO-FA historical solver | 200 | 0.75 | 10 | 0.3873 | 0.3805 | 0.3469 | 0.4773 |
| SRO-FA historical OOS | 200 | 0.75 | 10 | 0.1097 | 0.1128 | 0.0951 | 0.1270 |
| SRO-FA historical total | 200 | 0.75 | 10 | 1.1575 | 1.1569 | 1.0592 | 1.3576 |
| SRO-FA historical partition | 200 | 0.8 | 10 | 0.0085 | 0.0085 | 0.0076 | 0.0092 |
| SRO-FA historical solver | 200 | 0.8 | 10 | 0.4234 | 0.4141 | 0.3630 | 0.5203 |
| SRO-FA historical OOS | 200 | 0.8 | 10 | 0.1012 | 0.1002 | 0.0928 | 0.1133 |
| SRO-FA historical total | 200 | 0.8 | 10 | 1.2150 | 1.2392 | 1.0944 | 1.3252 |
| SRO-FA historical partition | 200 | 1.0 | 10 | 0.0081 | 0.0080 | 0.0055 | 0.0098 |
| SRO-FA historical solver | 200 | 1.0 | 10 | 0.5985 | 0.6018 | 0.5191 | 0.6654 |
| SRO-FA historical OOS | 200 | 1.0 | 10 | 0.1101 | 0.1106 | 0.0946 | 0.1232 |
| SRO-FA historical total | 200 | 1.0 | 10 | 1.6339 | 1.6404 | 1.5488 | 1.7198 |
| SRO-FA historical partition | 200 | 1.5 | 10 | 0.0083 | 0.0082 | 0.0064 | 0.0094 |
| SRO-FA historical solver | 200 | 1.5 | 10 | 1.1946 | 1.1713 | 1.0650 | 1.4023 |
| SRO-FA historical OOS | 200 | 1.5 | 10 | 0.1048 | 0.1037 | 0.0943 | 0.1197 |
| SRO-FA historical total | 200 | 1.5 | 10 | 2.9712 | 2.9482 | 2.7351 | 3.2608 |
| SCS nominal/full CCG (not certified) | 800 | all | 0 | unavailable | unavailable | unavailable | unavailable |
| SRO-FA historical partition | 800 | 0.001 | 10 | 0.0643 | 0.0652 | 0.0537 | 0.0740 |
| SRO-FA historical solver | 800 | 0.001 | 10 | 0.0609 | 0.0603 | 0.0482 | 0.0736 |
| SRO-FA historical OOS | 800 | 0.001 | 10 | 0.4192 | 0.4112 | 0.3876 | 0.4652 |
| SRO-FA historical total | 800 | 0.001 | 10 | 0.8880 | 0.8960 | 0.8299 | 0.9517 |
| SRO-FA historical partition | 800 | 0.005 | 10 | 0.0633 | 0.0628 | 0.0595 | 0.0682 |
| SRO-FA historical solver | 800 | 0.005 | 10 | 0.0735 | 0.0723 | 0.0607 | 0.0903 |
| SRO-FA historical OOS | 800 | 0.005 | 10 | 0.4237 | 0.4289 | 0.3926 | 0.4461 |
| SRO-FA historical total | 800 | 0.005 | 10 | 0.9292 | 0.9275 | 0.8867 | 0.9945 |
| SRO-FA historical partition | 800 | 0.01 | 10 | 0.0625 | 0.0636 | 0.0531 | 0.0722 |
| SRO-FA historical solver | 800 | 0.01 | 10 | 0.0808 | 0.0812 | 0.0703 | 0.0959 |
| SRO-FA historical OOS | 800 | 0.01 | 10 | 0.4081 | 0.4063 | 0.3734 | 0.4384 |
| SRO-FA historical total | 800 | 0.01 | 10 | 0.9467 | 0.9519 | 0.8949 | 1.0277 |
| SRO-FA historical partition | 800 | 0.02 | 10 | 0.0655 | 0.0663 | 0.0522 | 0.0784 |
| SRO-FA historical solver | 800 | 0.02 | 10 | 0.1045 | 0.1043 | 0.0946 | 0.1246 |
| SRO-FA historical OOS | 800 | 0.02 | 10 | 0.4144 | 0.4028 | 0.3857 | 0.4832 |
| SRO-FA historical total | 800 | 0.02 | 10 | 1.0110 | 1.0062 | 0.9337 | 1.1535 |
| SRO-FA historical partition | 800 | 0.05 | 10 | 0.0596 | 0.0603 | 0.0532 | 0.0650 |
| SRO-FA historical solver | 800 | 0.05 | 10 | 0.1718 | 0.1690 | 0.1521 | 0.2005 |
| SRO-FA historical OOS | 800 | 0.05 | 10 | 0.4033 | 0.4046 | 0.3818 | 0.4137 |
| SRO-FA historical total | 800 | 0.05 | 10 | 1.1594 | 1.1559 | 1.1123 | 1.2306 |
| SRO-FA historical partition | 800 | 0.1 | 10 | 0.0645 | 0.0640 | 0.0571 | 0.0730 |
| SRO-FA historical solver | 800 | 0.1 | 10 | 0.2891 | 0.2882 | 0.2304 | 0.3333 |
| SRO-FA historical OOS | 800 | 0.1 | 10 | 0.4070 | 0.4064 | 0.3681 | 0.4416 |
| SRO-FA historical total | 800 | 0.1 | 10 | 1.4176 | 1.4042 | 1.3487 | 1.5316 |
| SRO-FA historical partition | 800 | 0.15 | 10 | 0.0628 | 0.0609 | 0.0487 | 0.0805 |
| SRO-FA historical solver | 800 | 0.15 | 10 | 0.4767 | 0.4744 | 0.4262 | 0.5380 |
| SRO-FA historical OOS | 800 | 0.15 | 10 | 0.3994 | 0.3944 | 0.3693 | 0.4383 |
| SRO-FA historical total | 800 | 0.15 | 10 | 1.7615 | 1.7589 | 1.6920 | 1.8462 |
| SRO-FA historical partition | 800 | 0.2 | 10 | 0.0612 | 0.0607 | 0.0555 | 0.0719 |
| SRO-FA historical solver | 800 | 0.2 | 10 | 0.6857 | 0.6694 | 0.6289 | 0.7607 |
| SRO-FA historical OOS | 800 | 0.2 | 10 | 0.4251 | 0.4243 | 0.3916 | 0.4667 |
| SRO-FA historical total | 800 | 0.2 | 10 | 2.1613 | 2.1550 | 1.9939 | 2.3010 |
| SRO-FA historical partition | 800 | 0.3 | 10 | 0.0600 | 0.0611 | 0.0502 | 0.0736 |
| SRO-FA historical solver | 800 | 0.3 | 10 | 1.6679 | 1.6569 | 1.5566 | 1.8284 |
| SRO-FA historical OOS | 800 | 0.3 | 10 | 0.4102 | 0.4149 | 0.3738 | 0.4321 |
| SRO-FA historical total | 800 | 0.3 | 10 | 3.5811 | 3.5552 | 3.4230 | 3.7963 |
| SRO-FA historical partition | 800 | 0.4 | 10 | 0.0636 | 0.0623 | 0.0540 | 0.0745 |
| SRO-FA historical solver | 800 | 0.4 | 10 | 1.7974 | 1.7519 | 0.9235 | 2.6210 |
| SRO-FA historical OOS | 800 | 0.4 | 10 | 0.4084 | 0.4057 | 0.3728 | 0.4482 |
| SRO-FA historical total | 800 | 0.4 | 10 | 4.2293 | 4.2550 | 3.4449 | 5.0385 |
| SRO-FA historical partition | 800 | 0.45 | 10 | 0.0611 | 0.0593 | 0.0539 | 0.0806 |
| SRO-FA historical solver | 800 | 0.45 | 10 | 1.2369 | 1.2243 | 1.1254 | 1.3860 |
| SRO-FA historical OOS | 800 | 0.45 | 10 | 0.4009 | 0.3984 | 0.3689 | 0.4510 |
| SRO-FA historical total | 800 | 0.45 | 10 | 3.9488 | 3.8928 | 3.7585 | 4.2155 |
| SRO-FA historical partition | 800 | 0.5 | 10 | 0.0656 | 0.0643 | 0.0512 | 0.0846 |
| SRO-FA historical solver | 800 | 0.5 | 10 | 1.6279 | 1.5972 | 1.4341 | 1.8380 |
| SRO-FA historical OOS | 800 | 0.5 | 10 | 0.4012 | 0.4082 | 0.3676 | 0.4261 |
| SRO-FA historical total | 800 | 0.5 | 10 | 4.7422 | 4.6275 | 4.3808 | 5.6457 |
| SRO-FA historical partition | 800 | 0.6 | 10 | 0.0657 | 0.0667 | 0.0573 | 0.0745 |
| SRO-FA historical solver | 800 | 0.6 | 10 | 2.4715 | 2.5068 | 2.0011 | 2.8135 |
| SRO-FA historical OOS | 800 | 0.6 | 10 | 0.4107 | 0.3982 | 0.3531 | 0.4786 |
| SRO-FA historical total | 800 | 0.6 | 10 | 6.4247 | 6.6017 | 5.5912 | 6.9700 |
| SRO-FA historical partition | 800 | 0.75 | 10 | 0.0672 | 0.0671 | 0.0523 | 0.0873 |
| SRO-FA historical solver | 800 | 0.75 | 10 | 4.9291 | 5.0115 | 4.2840 | 5.6100 |
| SRO-FA historical OOS | 800 | 0.75 | 10 | 0.4002 | 0.3982 | 0.3431 | 0.4874 |
| SRO-FA historical total | 800 | 0.75 | 10 | 10.2732 | 10.4032 | 9.3671 | 11.1468 |
| SRO-FA historical partition | 800 | 0.8 | 10 | 0.0695 | 0.0688 | 0.0559 | 0.0886 |
| SRO-FA historical solver | 800 | 0.8 | 10 | 6.2571 | 6.3125 | 5.4720 | 7.0493 |
| SRO-FA historical OOS | 800 | 0.8 | 10 | 0.3926 | 0.3853 | 0.3526 | 0.4611 |
| SRO-FA historical total | 800 | 0.8 | 10 | 12.1156 | 12.1858 | 11.1155 | 13.1444 |
| SRO-FA historical partition | 800 | 1.0 | 10 | 0.0647 | 0.0619 | 0.0501 | 0.0870 |
| SRO-FA historical solver | 800 | 1.0 | 10 | 14.1321 | 14.0729 | 13.0003 | 15.2167 |
| SRO-FA historical OOS | 800 | 1.0 | 10 | 0.3933 | 0.3898 | 0.3525 | 0.4369 |
| SRO-FA historical total | 800 | 1.0 | 10 | 22.5453 | 22.5760 | 21.4497 | 23.7457 |
| SRO-FA historical partition | 800 | 1.5 | 10 | 0.0661 | 0.0645 | 0.0577 | 0.0797 |
| SRO-FA historical solver | 800 | 1.5 | 10 | 66.4912 | 66.9387 | 57.0304 | 73.5297 |
| SRO-FA historical OOS | 800 | 1.5 | 10 | 0.3848 | 0.3747 | 0.3513 | 0.4771 |
| SRO-FA historical total | 800 | 1.5 | 10 | 84.1212 | 84.7187 | 74.0978 | 91.4053 |
| Green N50 nominal/CCG (not established here) | 50 | all | 0 | unavailable | unavailable | unavailable | unavailable |
| SAA-kNN historical selection/CV (not established here) | 50/200/800 | all | 0 | unavailable | unavailable | unavailable | unavailable |

SRO-FA totals/partition/evaluation use historical time.time; time_lp is saved solver time,
not full optimization wrapper elapsed. No historical strict-status certificate is invented.
Original cache runtime fields, SRO seed records, nominal epochs, and AP/MP iteration times are retained
in the manifest and its frozen inputs. No new training, benchmark construction, or CV was performed.

## Outputs

Published figures: []; plot data: None.
runtime_aggregates.json contains current and persisted historical timings separately.
summary.json contains compact per-policy result paths and phase clocks; full policies are not rewritten per result.

SCS membership: 243358 positive residual observations; 348 above 1e-9; maximum 7.15170732288e-06.

End-to-end including reporting: 92.853639 s (92.853639 s / 1.547561 min / 0.025793 h); final timing/artifact serialization excluded.
