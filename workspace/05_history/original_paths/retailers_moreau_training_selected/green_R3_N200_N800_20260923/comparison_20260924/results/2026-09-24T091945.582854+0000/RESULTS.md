# Fixed-policy R3 comparison

Status: **completed**.
No new training, CV, or test-based selection. Green training counts 40/160/640; SCS 50/200/800.
Fresh strict green N200/N800 (180 policies) and SCS N50/N200/N800 (600 policies).
Green N50 (90 policies), SAA-kNN, and SRO-FA are cached historical references.
OOS: fixed seed 9999, 10,000 float64 samples. Previously inspected test: descriptive comparison.
IS: mean of per-center maxima on the returned checkpoint finite pool, not an adversarial supremum.
Green membership gate remains 1e-9. SCS saved coordinates are unchanged and not box-certified.

Host: gold-srv2.iem.technion.ac.il; affinity CPUs: 72; spawn workers: 32; one numerical/solver thread each.
Preflight: /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/preflight_v3/summary.json; SHA256: 5f123a8b0181d7b50ffbe8e150b4d0d45e29afa8030405cc872b105acb134b8a.
Manifest: frozen/manifest.json; SHA256: 482f729123d52ad4c547817f309372c8d3ad00cc18bd63f4d7171e7925ffebcd.
Final source hashes verified: True.
Current pipeline elapsed before reporting: 341.400226 s / 5.690004 min / 0.094833 h (341.400226 s).
Raw current times: policies/*/result.json; journals: policies/*/{oos,is}_solves.jsonl.
Per-model backend wall times exclude final result serialization; overall sweep includes dispatch and worker I/O.
Training references and AP/MP iteration timings remain in the checksummed frozen original summaries.

## Current evaluation wall times (seconds)

| Method | N | Radius | Status | Scope | n | Mean | Median | Min | Max |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| green | 200 | 0 | completed | is | 10 | 0.1060 | 0.1069 | 0.0843 | 0.1296 |
| green | 200 | 0 | completed | oos | 10 | 5.0251 | 5.0040 | 4.9119 | 5.2317 |
| green | 200 | 0 | completed | total | 10 | 8.5517 | 8.2299 | 7.8097 | 10.0412 |
| green | 200 | 0.001 | completed | is | 10 | 0.1132 | 0.1087 | 0.0941 | 0.1341 |
| green | 200 | 0.001 | completed | oos | 10 | 5.0832 | 5.0629 | 4.9705 | 5.2365 |
| green | 200 | 0.001 | completed | total | 10 | 8.6556 | 8.5906 | 8.0701 | 9.6727 |
| green | 200 | 0.003 | completed | is | 10 | 0.1096 | 0.1066 | 0.0950 | 0.1553 |
| green | 200 | 0.003 | completed | oos | 10 | 5.0436 | 5.0401 | 4.9502 | 5.1393 |
| green | 200 | 0.003 | completed | total | 10 | 8.6005 | 8.3681 | 8.1193 | 9.2668 |
| green | 200 | 0.01 | completed | is | 10 | 0.1003 | 0.0963 | 0.0841 | 0.1202 |
| green | 200 | 0.01 | completed | oos | 10 | 5.1266 | 5.0485 | 4.9518 | 5.4200 |
| green | 200 | 0.01 | completed | total | 10 | 8.7760 | 8.5469 | 7.8720 | 10.0504 |
| green | 200 | 0.03 | completed | is | 10 | 0.1113 | 0.1075 | 0.0951 | 0.1332 |
| green | 200 | 0.03 | completed | oos | 10 | 5.1000 | 5.1122 | 4.9498 | 5.2551 |
| green | 200 | 0.03 | completed | total | 10 | 8.6453 | 8.5121 | 8.0165 | 9.6749 |
| green | 200 | 0.2 | completed | is | 10 | 0.2362 | 0.2393 | 0.1648 | 0.2975 |
| green | 200 | 0.2 | completed | oos | 10 | 5.1080 | 5.0962 | 5.0097 | 5.2786 |
| green | 200 | 0.2 | completed | total | 10 | 8.7276 | 8.6764 | 8.1770 | 9.3410 |
| green | 200 | 0.4 | completed | is | 10 | 0.2824 | 0.2821 | 0.2382 | 0.3183 |
| green | 200 | 0.4 | completed | oos | 10 | 5.0644 | 5.0699 | 4.9906 | 5.1458 |
| green | 200 | 0.4 | completed | total | 10 | 8.7573 | 8.6224 | 8.2376 | 9.8973 |
| green | 200 | 0.7 | completed | is | 10 | 0.3684 | 0.3727 | 0.2971 | 0.4712 |
| green | 200 | 0.7 | completed | oos | 10 | 5.0923 | 5.0795 | 4.9421 | 5.2994 |
| green | 200 | 0.7 | completed | total | 10 | 8.9098 | 8.6211 | 8.3665 | 9.9456 |
| green | 200 | 1.5 | completed | is | 10 | 0.4255 | 0.4377 | 0.3090 | 0.5475 |
| green | 200 | 1.5 | completed | oos | 10 | 5.0323 | 5.0430 | 4.9047 | 5.1278 |
| green | 200 | 1.5 | completed | total | 10 | 8.9312 | 8.5876 | 8.3691 | 10.1384 |
| green | 800 | 0 | completed | is | 10 | 0.3555 | 0.3527 | 0.3382 | 0.3860 |
| green | 800 | 0 | completed | oos | 10 | 5.0805 | 5.1010 | 4.9040 | 5.1987 |
| green | 800 | 0 | completed | total | 10 | 8.5000 | 8.4671 | 8.1511 | 8.9202 |
| green | 800 | 0.001 | completed | is | 10 | 0.3644 | 0.3602 | 0.3462 | 0.3952 |
| green | 800 | 0.001 | completed | oos | 10 | 5.1035 | 5.0668 | 5.0047 | 5.4942 |
| green | 800 | 0.001 | completed | total | 10 | 8.4854 | 8.4421 | 8.2256 | 8.9762 |
| green | 800 | 0.003 | completed | is | 10 | 0.3616 | 0.3609 | 0.3363 | 0.4218 |
| green | 800 | 0.003 | completed | oos | 10 | 5.0644 | 5.0467 | 4.9583 | 5.1809 |
| green | 800 | 0.003 | completed | total | 10 | 8.4941 | 8.4259 | 8.2945 | 8.8004 |
| green | 800 | 0.01 | completed | is | 10 | 0.3611 | 0.3530 | 0.3361 | 0.4271 |
| green | 800 | 0.01 | completed | oos | 10 | 5.0556 | 5.0400 | 4.9960 | 5.1481 |
| green | 800 | 0.01 | completed | total | 10 | 8.3921 | 8.3421 | 8.1682 | 8.6092 |
| green | 800 | 0.03 | completed | is | 10 | 0.3604 | 0.3604 | 0.3368 | 0.3911 |
| green | 800 | 0.03 | completed | oos | 10 | 5.0636 | 5.0336 | 4.8710 | 5.2905 |
| green | 800 | 0.03 | completed | total | 10 | 8.4363 | 8.3601 | 8.0214 | 9.1590 |
| green | 800 | 0.2 | completed | is | 10 | 0.6576 | 0.6644 | 0.3477 | 0.8720 |
| green | 800 | 0.2 | completed | oos | 10 | 5.0318 | 5.0273 | 4.8883 | 5.1599 |
| green | 800 | 0.2 | completed | total | 10 | 8.8027 | 8.8956 | 8.3330 | 9.2717 |
| green | 800 | 0.4 | completed | is | 10 | 0.9474 | 0.8907 | 0.8296 | 1.2098 |
| green | 800 | 0.4 | completed | oos | 10 | 5.0535 | 5.0473 | 4.9183 | 5.2067 |
| green | 800 | 0.4 | completed | total | 10 | 9.0297 | 8.9632 | 8.6803 | 9.5460 |
| green | 800 | 0.7 | completed | is | 10 | 1.1011 | 1.1580 | 0.8698 | 1.2750 |
| green | 800 | 0.7 | completed | oos | 10 | 5.0554 | 5.0089 | 4.9749 | 5.2894 |
| green | 800 | 0.7 | completed | total | 10 | 9.2166 | 9.2192 | 8.8039 | 9.6113 |
| green | 800 | 1.5 | completed | is | 10 | 1.3149 | 1.3287 | 1.1419 | 1.4560 |
| green | 800 | 1.5 | completed | oos | 10 | 5.0131 | 4.9855 | 4.8928 | 5.1738 |
| green | 800 | 1.5 | completed | total | 10 | 9.4405 | 9.4802 | 9.0213 | 9.7938 |
| scs | 50 | 0 | completed | is | 10 | 0.0502 | 0.0460 | 0.0341 | 0.0845 |
| scs | 50 | 0 | completed | oos | 10 | 5.0504 | 5.0374 | 4.9685 | 5.1556 |
| scs | 50 | 0 | completed | total | 10 | 7.9803 | 7.9297 | 7.7043 | 8.3707 |
| scs | 50 | 0.001 | completed | is | 10 | 0.0756 | 0.0761 | 0.0342 | 0.1206 |
| scs | 50 | 0.001 | completed | oos | 10 | 5.0896 | 5.0495 | 4.9861 | 5.2409 |
| scs | 50 | 0.001 | completed | total | 10 | 8.3331 | 8.2462 | 8.0820 | 8.8176 |
| scs | 50 | 0.0015 | completed | is | 10 | 0.0749 | 0.0827 | 0.0339 | 0.1178 |
| scs | 50 | 0.0015 | completed | oos | 10 | 5.0617 | 5.0483 | 4.9888 | 5.1746 |
| scs | 50 | 0.0015 | completed | total | 10 | 8.2893 | 8.2895 | 7.8877 | 8.5736 |
| scs | 50 | 0.002 | completed | is | 10 | 0.0735 | 0.0766 | 0.0338 | 0.0978 |
| scs | 50 | 0.002 | completed | oos | 10 | 5.0378 | 5.0193 | 4.8956 | 5.2434 |
| scs | 50 | 0.002 | completed | total | 10 | 8.1897 | 8.1794 | 7.6358 | 8.4810 |
| scs | 50 | 0.003 | completed | is | 10 | 0.1279 | 0.0861 | 0.0341 | 0.3624 |
| scs | 50 | 0.003 | completed | oos | 10 | 5.0663 | 5.0655 | 5.0010 | 5.1600 |
| scs | 50 | 0.003 | completed | total | 10 | 8.0435 | 8.0379 | 7.7263 | 8.4031 |
| scs | 50 | 0.005 | completed | is | 10 | 0.0773 | 0.0820 | 0.0340 | 0.1210 |
| scs | 50 | 0.005 | completed | oos | 10 | 5.0538 | 5.0509 | 4.9288 | 5.2933 |
| scs | 50 | 0.005 | completed | total | 10 | 8.0474 | 7.9485 | 7.7651 | 8.5160 |
| scs | 50 | 0.01 | completed | is | 10 | 0.0786 | 0.0785 | 0.0470 | 0.0973 |
| scs | 50 | 0.01 | completed | oos | 10 | 5.0533 | 5.0003 | 4.8403 | 5.4202 |
| scs | 50 | 0.01 | completed | total | 10 | 8.0657 | 8.0177 | 7.7036 | 8.6711 |
| scs | 50 | 0.02 | completed | is | 10 | 0.0722 | 0.0775 | 0.0337 | 0.0941 |
| scs | 50 | 0.02 | completed | oos | 10 | 5.1950 | 5.1308 | 5.0217 | 5.5605 |
| scs | 50 | 0.02 | completed | total | 10 | 8.2145 | 8.1704 | 7.8622 | 8.7677 |
| scs | 50 | 0.05 | completed | is | 10 | 0.0727 | 0.0770 | 0.0336 | 0.1138 |
| scs | 50 | 0.05 | completed | oos | 10 | 5.0253 | 5.0161 | 4.9185 | 5.1623 |
| scs | 50 | 0.05 | completed | total | 10 | 8.2000 | 8.1669 | 7.8195 | 8.7551 |
| scs | 50 | 0.1 | completed | is | 10 | 0.0762 | 0.0668 | 0.0461 | 0.1451 |
| scs | 50 | 0.1 | completed | oos | 10 | 5.0986 | 5.0714 | 4.9765 | 5.2701 |
| scs | 50 | 0.1 | completed | total | 10 | 8.2696 | 8.2495 | 8.0324 | 8.6571 |
| scs | 50 | 0.15 | completed | is | 10 | 0.0909 | 0.0872 | 0.0460 | 0.1589 |
| scs | 50 | 0.15 | completed | oos | 10 | 5.1353 | 5.0750 | 5.0093 | 5.4185 |
| scs | 50 | 0.15 | completed | total | 10 | 8.1492 | 8.1340 | 7.7731 | 8.6787 |
| scs | 50 | 0.2 | completed | is | 10 | 0.1249 | 0.1189 | 0.0777 | 0.2127 |
| scs | 50 | 0.2 | completed | oos | 10 | 5.0544 | 5.0217 | 4.9790 | 5.2117 |
| scs | 50 | 0.2 | completed | total | 10 | 8.1937 | 8.1855 | 7.9421 | 8.4108 |
| scs | 50 | 0.3 | completed | is | 10 | 0.1271 | 0.1271 | 0.0847 | 0.1635 |
| scs | 50 | 0.3 | completed | oos | 10 | 5.0783 | 5.0753 | 4.9369 | 5.2007 |
| scs | 50 | 0.3 | completed | total | 10 | 8.2384 | 8.1780 | 7.9855 | 8.8414 |
| scs | 50 | 0.4 | completed | is | 10 | 0.1670 | 0.1628 | 0.1351 | 0.2182 |
| scs | 50 | 0.4 | completed | oos | 10 | 5.1268 | 5.1002 | 4.9910 | 5.4365 |
| scs | 50 | 0.4 | completed | total | 10 | 8.2963 | 8.3469 | 7.9227 | 8.6883 |
| scs | 50 | 0.45 | completed | is | 10 | 0.1729 | 0.1751 | 0.1148 | 0.2300 |
| scs | 50 | 0.45 | completed | oos | 10 | 5.1074 | 5.0120 | 4.9083 | 5.4448 |
| scs | 50 | 0.45 | completed | total | 10 | 8.2586 | 8.2545 | 7.9291 | 8.7571 |
| scs | 50 | 0.5 | completed | is | 10 | 0.2030 | 0.2020 | 0.1448 | 0.2672 |
| scs | 50 | 0.5 | completed | oos | 10 | 5.2890 | 5.3123 | 4.9661 | 5.8338 |
| scs | 50 | 0.5 | completed | total | 10 | 8.5378 | 8.5532 | 8.0822 | 8.8595 |
| scs | 50 | 0.55 | completed | is | 10 | 0.1966 | 0.1867 | 0.1352 | 0.3369 |
| scs | 50 | 0.55 | completed | oos | 10 | 5.1410 | 5.1119 | 4.9989 | 5.4383 |
| scs | 50 | 0.55 | completed | total | 10 | 8.2085 | 8.2048 | 7.8094 | 8.5837 |
| scs | 50 | 0.6 | completed | is | 10 | 0.2172 | 0.2222 | 0.1352 | 0.2791 |
| scs | 50 | 0.6 | completed | oos | 10 | 5.0290 | 4.9934 | 4.9291 | 5.1657 |
| scs | 50 | 0.6 | completed | total | 10 | 8.1116 | 8.1693 | 7.7734 | 8.4101 |
| scs | 50 | 0.65 | completed | is | 10 | 0.2163 | 0.2213 | 0.1352 | 0.2661 |
| scs | 50 | 0.65 | completed | oos | 10 | 5.0698 | 5.0575 | 4.9569 | 5.2771 |
| scs | 50 | 0.65 | completed | total | 10 | 8.2708 | 8.2488 | 8.0056 | 8.6233 |
| scs | 50 | 0.7 | completed | is | 10 | 0.2106 | 0.1969 | 0.1278 | 0.2854 |
| scs | 50 | 0.7 | completed | oos | 10 | 5.0846 | 5.0586 | 4.9327 | 5.3212 |
| scs | 50 | 0.7 | completed | total | 10 | 8.2855 | 8.2628 | 7.9690 | 8.5799 |
| scs | 50 | 0.75 | completed | is | 10 | 0.2430 | 0.2412 | 0.1966 | 0.2862 |
| scs | 50 | 0.75 | completed | oos | 10 | 5.0738 | 5.0740 | 4.8916 | 5.3389 |
| scs | 50 | 0.75 | completed | total | 10 | 8.3712 | 8.3160 | 8.1979 | 8.7286 |
| scs | 50 | 0.8 | completed | is | 10 | 0.2352 | 0.2378 | 0.1351 | 0.2968 |
| scs | 50 | 0.8 | completed | oos | 10 | 5.0363 | 5.0384 | 4.9469 | 5.1420 |
| scs | 50 | 0.8 | completed | total | 10 | 8.3754 | 8.3581 | 7.9882 | 8.8811 |
| scs | 50 | 1 | completed | is | 10 | 0.2222 | 0.2030 | 0.1552 | 0.3323 |
| scs | 50 | 1 | completed | oos | 10 | 5.0404 | 5.0678 | 4.9065 | 5.1390 |
| scs | 50 | 1 | completed | total | 10 | 8.2498 | 8.2346 | 7.9674 | 8.6046 |
| scs | 50 | 1.5 | completed | is | 10 | 0.2033 | 0.2073 | 0.1601 | 0.2460 |
| scs | 50 | 1.5 | completed | oos | 10 | 4.9811 | 5.0300 | 4.5993 | 5.2271 |
| scs | 50 | 1.5 | completed | total | 10 | 8.0368 | 7.9148 | 7.7881 | 8.5560 |
| scs | 200 | 0 | completed | is | 10 | 0.1300 | 0.1268 | 0.1049 | 0.1643 |
| scs | 200 | 0 | completed | oos | 10 | 5.0122 | 4.9991 | 4.9509 | 5.1211 |
| scs | 200 | 0 | completed | total | 10 | 8.1479 | 8.1357 | 7.8795 | 8.4886 |
| scs | 200 | 0.001 | completed | is | 10 | 0.2911 | 0.3163 | 0.2051 | 0.3512 |
| scs | 200 | 0.001 | completed | oos | 10 | 5.0501 | 5.0256 | 4.9733 | 5.1581 |
| scs | 200 | 0.001 | completed | total | 10 | 8.2909 | 8.2833 | 7.9774 | 8.5739 |
| scs | 200 | 0.005 | completed | is | 10 | 0.3108 | 0.3158 | 0.2291 | 0.4408 |
| scs | 200 | 0.005 | completed | oos | 10 | 5.1047 | 5.0868 | 4.9679 | 5.3221 |
| scs | 200 | 0.005 | completed | total | 10 | 8.3876 | 8.3887 | 8.0734 | 8.6220 |
| scs | 200 | 0.01 | completed | is | 10 | 0.2883 | 0.3021 | 0.2173 | 0.3505 |
| scs | 200 | 0.01 | completed | oos | 10 | 5.0100 | 5.0205 | 4.9379 | 5.1348 |
| scs | 200 | 0.01 | completed | total | 10 | 8.2908 | 8.2356 | 7.9158 | 8.7954 |
| scs | 200 | 0.02 | completed | is | 10 | 0.2540 | 0.2272 | 0.2155 | 0.3388 |
| scs | 200 | 0.02 | completed | oos | 10 | 5.0294 | 5.0264 | 4.9632 | 5.1137 |
| scs | 200 | 0.02 | completed | total | 10 | 8.2888 | 8.2977 | 8.0235 | 8.5445 |
| scs | 200 | 0.05 | completed | is | 10 | 0.2305 | 0.2285 | 0.2154 | 0.2524 |
| scs | 200 | 0.05 | completed | oos | 10 | 5.0420 | 5.0067 | 4.9263 | 5.2080 |
| scs | 200 | 0.05 | completed | total | 10 | 8.2482 | 8.2522 | 8.1086 | 8.3969 |
| scs | 200 | 0.1 | completed | is | 10 | 0.2432 | 0.2426 | 0.2170 | 0.2641 |
| scs | 200 | 0.1 | completed | oos | 10 | 5.0294 | 5.0151 | 4.9126 | 5.2534 |
| scs | 200 | 0.1 | completed | total | 10 | 8.2172 | 8.1758 | 8.0082 | 8.5726 |
| scs | 200 | 0.15 | completed | is | 10 | 0.2450 | 0.2284 | 0.1276 | 0.3287 |
| scs | 200 | 0.15 | completed | oos | 10 | 5.0333 | 5.0246 | 4.9118 | 5.1751 |
| scs | 200 | 0.15 | completed | total | 10 | 8.3118 | 8.2684 | 7.9579 | 8.6708 |
| scs | 200 | 0.2 | completed | is | 10 | 0.3015 | 0.3227 | 0.2157 | 0.3892 |
| scs | 200 | 0.2 | completed | oos | 10 | 5.0186 | 5.0241 | 4.9050 | 5.1407 |
| scs | 200 | 0.2 | completed | total | 10 | 8.2654 | 8.2834 | 7.9023 | 8.5650 |
| scs | 200 | 0.3 | completed | is | 10 | 0.3394 | 0.3244 | 0.3169 | 0.4393 |
| scs | 200 | 0.3 | completed | oos | 10 | 5.1054 | 5.1017 | 4.9179 | 5.2896 |
| scs | 200 | 0.3 | completed | total | 10 | 8.3157 | 8.2949 | 7.9471 | 8.8124 |
| scs | 200 | 0.4 | completed | is | 10 | 0.4366 | 0.4288 | 0.3056 | 0.5707 |
| scs | 200 | 0.4 | completed | oos | 10 | 5.0383 | 4.9930 | 4.9513 | 5.2324 |
| scs | 200 | 0.4 | completed | total | 10 | 8.4876 | 8.4243 | 8.2657 | 8.9333 |
| scs | 200 | 0.45 | completed | is | 10 | 0.4466 | 0.4418 | 0.2284 | 0.6137 |
| scs | 200 | 0.45 | completed | oos | 10 | 5.1204 | 5.1053 | 4.9587 | 5.3448 |
| scs | 200 | 0.45 | completed | total | 10 | 8.6084 | 8.6016 | 8.0948 | 9.0384 |
| scs | 200 | 0.5 | completed | is | 10 | 0.4978 | 0.5197 | 0.4077 | 0.5704 |
| scs | 200 | 0.5 | completed | oos | 10 | 5.0666 | 5.0235 | 4.9417 | 5.2238 |
| scs | 200 | 0.5 | completed | total | 10 | 8.4304 | 8.4382 | 8.0722 | 8.7609 |
| scs | 200 | 0.6 | completed | is | 10 | 0.5385 | 0.5253 | 0.5073 | 0.6340 |
| scs | 200 | 0.6 | completed | oos | 10 | 5.0275 | 5.0305 | 4.9395 | 5.1518 |
| scs | 200 | 0.6 | completed | total | 10 | 8.5481 | 8.5388 | 8.2777 | 8.7699 |
| scs | 200 | 0.75 | completed | is | 10 | 0.5475 | 0.5409 | 0.3474 | 0.6943 |
| scs | 200 | 0.75 | completed | oos | 10 | 5.0681 | 5.0773 | 4.8902 | 5.2625 |
| scs | 200 | 0.75 | completed | total | 10 | 8.5891 | 8.5569 | 8.2317 | 9.1966 |
| scs | 200 | 0.8 | completed | is | 10 | 0.6159 | 0.6202 | 0.4541 | 0.7663 |
| scs | 200 | 0.8 | completed | oos | 10 | 5.0879 | 5.0620 | 4.9494 | 5.2441 |
| scs | 200 | 0.8 | completed | total | 10 | 8.7367 | 8.7068 | 8.4369 | 9.1218 |
| scs | 200 | 1 | completed | is | 10 | 0.5563 | 0.5534 | 0.4277 | 0.6414 |
| scs | 200 | 1 | completed | oos | 10 | 5.0455 | 5.0481 | 4.9500 | 5.1594 |
| scs | 200 | 1 | completed | total | 10 | 8.5334 | 8.5710 | 8.1049 | 8.8560 |
| scs | 200 | 1.5 | completed | is | 10 | 0.5455 | 0.5382 | 0.4983 | 0.6085 |
| scs | 200 | 1.5 | completed | oos | 10 | 5.0112 | 4.9901 | 4.8633 | 5.1651 |
| scs | 200 | 1.5 | completed | total | 10 | 8.5816 | 8.5602 | 8.1390 | 8.9673 |
| scs | 800 | 0 | completed | is | 10 | 0.4478 | 0.4401 | 0.4204 | 0.5011 |
| scs | 800 | 0 | completed | oos | 10 | 5.1138 | 5.1054 | 4.9609 | 5.2644 |
| scs | 800 | 0 | completed | total | 10 | 8.5297 | 8.4561 | 8.1706 | 9.1653 |
| scs | 800 | 0.001 | completed | is | 10 | 0.8446 | 0.8393 | 0.8145 | 0.8791 |
| scs | 800 | 0.001 | completed | oos | 10 | 5.0810 | 5.1053 | 4.8506 | 5.1891 |
| scs | 800 | 0.001 | completed | total | 10 | 8.8983 | 8.9097 | 8.4375 | 9.1946 |
| scs | 800 | 0.005 | completed | is | 10 | 0.8309 | 0.8323 | 0.8006 | 0.8470 |
| scs | 800 | 0.005 | completed | oos | 10 | 5.0477 | 5.0402 | 4.9397 | 5.2299 |
| scs | 800 | 0.005 | completed | total | 10 | 8.8708 | 8.8479 | 8.6488 | 9.3825 |
| scs | 800 | 0.01 | completed | is | 10 | 0.8021 | 0.8360 | 0.4538 | 0.8555 |
| scs | 800 | 0.01 | completed | oos | 10 | 5.0506 | 5.0413 | 4.9410 | 5.1540 |
| scs | 800 | 0.01 | completed | total | 10 | 8.7862 | 8.7879 | 8.3751 | 9.1082 |
| scs | 800 | 0.02 | completed | is | 10 | 0.8026 | 0.8377 | 0.4165 | 0.8938 |
| scs | 800 | 0.02 | completed | oos | 10 | 5.1106 | 5.0995 | 5.0395 | 5.2338 |
| scs | 800 | 0.02 | completed | total | 10 | 8.8856 | 8.8785 | 8.3269 | 9.2941 |
| scs | 800 | 0.05 | completed | is | 10 | 0.8127 | 0.8385 | 0.4077 | 0.9840 |
| scs | 800 | 0.05 | completed | oos | 10 | 5.0448 | 5.0388 | 4.9345 | 5.2209 |
| scs | 800 | 0.05 | completed | total | 10 | 8.9417 | 9.0565 | 8.1500 | 9.3180 |
| scs | 800 | 0.1 | completed | is | 10 | 0.8577 | 0.8405 | 0.8003 | 0.9732 |
| scs | 800 | 0.1 | completed | oos | 10 | 5.0512 | 5.0458 | 4.8584 | 5.2737 |
| scs | 800 | 0.1 | completed | total | 10 | 8.9923 | 9.0137 | 8.6952 | 9.2435 |
| scs | 800 | 0.15 | completed | is | 10 | 0.7669 | 0.8347 | 0.4286 | 0.8991 |
| scs | 800 | 0.15 | completed | oos | 10 | 5.1216 | 5.1203 | 4.9688 | 5.3412 |
| scs | 800 | 0.15 | completed | total | 10 | 8.8109 | 8.8140 | 8.4322 | 9.0936 |
| scs | 800 | 0.2 | completed | is | 10 | 0.8436 | 0.8462 | 0.4433 | 1.2485 |
| scs | 800 | 0.2 | completed | oos | 10 | 5.0213 | 5.0338 | 4.8810 | 5.0796 |
| scs | 800 | 0.2 | completed | total | 10 | 8.8997 | 8.9046 | 8.5234 | 9.3195 |
| scs | 800 | 0.3 | completed | is | 10 | 1.1821 | 1.2445 | 0.4199 | 1.3897 |
| scs | 800 | 0.3 | completed | oos | 10 | 5.0781 | 5.0858 | 4.9770 | 5.1723 |
| scs | 800 | 0.3 | completed | total | 10 | 9.3119 | 9.3428 | 8.2427 | 9.8778 |
| scs | 800 | 0.4 | completed | is | 10 | 1.3480 | 1.2539 | 1.2136 | 1.6959 |
| scs | 800 | 0.4 | completed | oos | 10 | 5.1005 | 5.0841 | 4.9566 | 5.2797 |
| scs | 800 | 0.4 | completed | total | 10 | 9.4814 | 9.4305 | 9.0593 | 10.0018 |
| scs | 800 | 0.45 | completed | is | 10 | 1.3718 | 1.2469 | 1.2255 | 1.6976 |
| scs | 800 | 0.45 | completed | oos | 10 | 5.1442 | 5.1119 | 5.0307 | 5.3668 |
| scs | 800 | 0.45 | completed | total | 10 | 9.6010 | 9.5139 | 9.2269 | 10.1175 |
| scs | 800 | 0.5 | completed | is | 10 | 1.5166 | 1.6171 | 1.2239 | 1.7276 |
| scs | 800 | 0.5 | completed | oos | 10 | 5.1285 | 5.1221 | 4.9893 | 5.4218 |
| scs | 800 | 0.5 | completed | total | 10 | 9.7953 | 9.9191 | 9.2575 | 10.1462 |
| scs | 800 | 0.6 | completed | is | 10 | 1.5648 | 1.6123 | 1.2617 | 1.7383 |
| scs | 800 | 0.6 | completed | oos | 10 | 5.1643 | 5.0873 | 5.0023 | 5.5963 |
| scs | 800 | 0.6 | completed | total | 10 | 9.7319 | 9.8085 | 9.4051 | 9.9433 |
| scs | 800 | 0.75 | completed | is | 10 | 1.7222 | 1.6570 | 1.6175 | 2.0543 |
| scs | 800 | 0.75 | completed | oos | 10 | 5.0564 | 5.0330 | 4.9775 | 5.2036 |
| scs | 800 | 0.75 | completed | total | 10 | 9.7386 | 9.5906 | 9.4301 | 10.1379 |
| scs | 800 | 0.8 | completed | is | 10 | 1.7212 | 1.6502 | 1.5912 | 2.1045 |
| scs | 800 | 0.8 | completed | oos | 10 | 5.0441 | 5.0540 | 4.9702 | 5.1549 |
| scs | 800 | 0.8 | completed | total | 10 | 9.7923 | 9.7588 | 9.4565 | 10.1623 |
| scs | 800 | 1 | completed | is | 10 | 1.7048 | 1.6433 | 1.5765 | 2.0514 |
| scs | 800 | 1 | completed | oos | 10 | 5.0913 | 5.0737 | 4.9844 | 5.2158 |
| scs | 800 | 1 | completed | total | 10 | 9.8844 | 9.7512 | 9.4563 | 10.4634 |
| scs | 800 | 1.5 | completed | is | 10 | 1.7784 | 1.6392 | 1.6011 | 2.0474 |
| scs | 800 | 1.5 | completed | oos | 10 | 4.9549 | 4.9565 | 4.8820 | 5.0208 |
| scs | 800 | 1.5 | completed | total | 10 | 9.6425 | 9.6046 | 9.3580 | 9.9435 |

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

Published figures: ['/data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/results/2026-09-24T091945.582854+0000/comparison_R3.png', '/data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/results/2026-09-24T091945.582854+0000/comparison_R3.pdf']; plot data: /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/results/2026-09-24T091945.582854+0000/plot_data.json.
runtime_aggregates.json contains current and persisted historical timings separately.
summary.json contains compact per-policy result paths and phase clocks; full policies are not rewritten per result.

SCS membership: 243358 positive residual observations; 348 above 1e-9; maximum 7.15170732288e-06.

End-to-end including reporting: 341.443545 s (341.443545 s / 5.690726 min / 0.094845 h); final timing/artifact serialization excluded.
