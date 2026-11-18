# Fixed-policy R10 comparison

Status: **completed**.
No new training, CV, or test-based selection. Green training counts 40/160/640; SCS 50/200/800.
Fresh strict green N50/N200/N800 (270 policies) and SCS N50/N200/N800 (540 policies).
Only SAA-kNN and SRO-FA are cached historical references; green N50 is fresh.
OOS: fixed seed 9999, 10,000 float64 samples. Previously inspected test: descriptive comparison.
IS: mean of per-center maxima on the returned checkpoint finite pool, not an adversarial supremum.
Green membership gate remains 1e-9. SCS saved coordinates are unchanged and not box-certified.

Host: gold-srv2.iem.technion.ac.il; affinity CPUs: 72; spawn workers: 32; one numerical/solver thread each.
Preflight: /data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/preflight_v1/summary.json; SHA256: efdd81596a4c851750faa1049868daa251e0e5101e9749575c06829a8d5acd40.
Manifest: frozen/manifest.json; SHA256: 9f201b3ce4b2a0b88f2fb1ecb003d85a948ffeae3f6d795b0c3b8e6662ee53ec.
Final source hashes verified: True.
Current pipeline elapsed before reporting: 925.208416 s / 15.420140 min / 0.257002 h (925.208416 s).
Raw current times: policies/*/result.json; journals: policies/*/{oos,is}_solves.jsonl.
Per-model backend wall times exclude final result serialization; overall sweep includes dispatch and worker I/O.
Training references and AP/MP iteration timings remain in the checksummed frozen original summaries.

## Current evaluation wall times (seconds)

| Method | N | Radius | Status | Scope | n | Mean | Median | Min | Max |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| green | 50 | 0 | completed | is | 10 | 0.0484 | 0.0468 | 0.0336 | 0.0799 |
| green | 50 | 0 | completed | oos | 10 | 6.0661 | 6.0363 | 5.9456 | 6.3413 |
| green | 50 | 0 | completed | total | 10 | 13.8448 | 13.8569 | 12.9317 | 15.0502 |
| green | 50 | 0.001 | completed | is | 10 | 0.0485 | 0.0462 | 0.0338 | 0.0829 |
| green | 50 | 0.001 | completed | oos | 10 | 6.0126 | 6.0036 | 5.9215 | 6.1457 |
| green | 50 | 0.001 | completed | total | 10 | 14.0402 | 14.0446 | 13.5324 | 14.9369 |
| green | 50 | 0.003 | completed | is | 10 | 0.0542 | 0.0581 | 0.0348 | 0.0799 |
| green | 50 | 0.003 | completed | oos | 10 | 6.0856 | 6.0515 | 5.9069 | 6.4663 |
| green | 50 | 0.003 | completed | total | 10 | 14.3113 | 13.9891 | 13.5090 | 15.7033 |
| green | 50 | 0.01 | completed | is | 10 | 0.0547 | 0.0500 | 0.0341 | 0.0841 |
| green | 50 | 0.01 | completed | oos | 10 | 6.0329 | 5.9654 | 5.8123 | 6.4515 |
| green | 50 | 0.01 | completed | total | 10 | 13.8466 | 13.6192 | 13.1105 | 15.1192 |
| green | 50 | 0.03 | completed | is | 10 | 0.0587 | 0.0521 | 0.0333 | 0.0975 |
| green | 50 | 0.03 | completed | oos | 10 | 5.9839 | 5.9441 | 5.8531 | 6.2743 |
| green | 50 | 0.03 | completed | total | 10 | 13.8821 | 13.7436 | 13.1164 | 15.5220 |
| green | 50 | 0.2 | completed | is | 10 | 0.2014 | 0.2010 | 0.1052 | 0.3098 |
| green | 50 | 0.2 | completed | oos | 10 | 5.9737 | 5.9632 | 5.8815 | 6.1567 |
| green | 50 | 0.2 | completed | total | 10 | 14.0849 | 13.7871 | 13.5823 | 15.2595 |
| green | 50 | 0.4 | completed | is | 10 | 0.3261 | 0.3220 | 0.1948 | 0.4794 |
| green | 50 | 0.4 | completed | oos | 10 | 6.0480 | 6.0445 | 5.8603 | 6.3791 |
| green | 50 | 0.4 | completed | total | 10 | 14.5309 | 14.5686 | 13.5837 | 15.5356 |
| green | 50 | 0.7 | completed | is | 10 | 0.4024 | 0.3803 | 0.2254 | 0.6486 |
| green | 50 | 0.7 | completed | oos | 10 | 5.9945 | 5.9935 | 5.8054 | 6.1513 |
| green | 50 | 0.7 | completed | total | 10 | 14.5727 | 14.5178 | 13.8226 | 15.5572 |
| green | 50 | 1.5 | completed | is | 10 | 0.3649 | 0.3802 | 0.2266 | 0.4668 |
| green | 50 | 1.5 | completed | oos | 10 | 5.9654 | 5.9833 | 5.7754 | 6.1299 |
| green | 50 | 1.5 | completed | total | 10 | 14.6677 | 14.6374 | 13.7445 | 15.4564 |
| green | 200 | 0 | completed | is | 10 | 0.1226 | 0.1223 | 0.1043 | 0.1551 |
| green | 200 | 0 | completed | oos | 10 | 5.9466 | 5.9096 | 5.8084 | 6.1181 |
| green | 200 | 0 | completed | total | 10 | 13.3870 | 13.3670 | 12.9879 | 13.8529 |
| green | 200 | 0.001 | completed | is | 10 | 0.1201 | 0.1169 | 0.1047 | 0.1410 |
| green | 200 | 0.001 | completed | oos | 10 | 5.9508 | 5.8916 | 5.8007 | 6.1920 |
| green | 200 | 0.001 | completed | total | 10 | 13.2866 | 13.2411 | 13.0030 | 13.7031 |
| green | 200 | 0.003 | completed | is | 10 | 0.1224 | 0.1166 | 0.1046 | 0.1759 |
| green | 200 | 0.003 | completed | oos | 10 | 5.9186 | 5.8811 | 5.7729 | 6.1013 |
| green | 200 | 0.003 | completed | total | 10 | 13.3528 | 13.2888 | 13.0541 | 13.9916 |
| green | 200 | 0.01 | completed | is | 10 | 0.1219 | 0.1173 | 0.1047 | 0.1397 |
| green | 200 | 0.01 | completed | oos | 10 | 5.9269 | 5.9362 | 5.8135 | 6.0275 |
| green | 200 | 0.01 | completed | total | 10 | 13.3571 | 13.2653 | 13.0764 | 13.8987 |
| green | 200 | 0.03 | completed | is | 10 | 0.1201 | 0.1170 | 0.1044 | 0.1530 |
| green | 200 | 0.03 | completed | oos | 10 | 5.9822 | 5.9537 | 5.8631 | 6.1652 |
| green | 200 | 0.03 | completed | total | 10 | 13.4582 | 13.4052 | 13.1639 | 13.8889 |
| green | 200 | 0.2 | completed | is | 10 | 0.3008 | 0.3030 | 0.1278 | 0.4192 |
| green | 200 | 0.2 | completed | oos | 10 | 5.8632 | 5.8605 | 5.7650 | 5.9555 |
| green | 200 | 0.2 | completed | total | 10 | 13.5697 | 13.5644 | 13.1297 | 14.1035 |
| green | 200 | 0.4 | completed | is | 10 | 0.4968 | 0.4941 | 0.3870 | 0.6304 |
| green | 200 | 0.4 | completed | oos | 10 | 6.0136 | 5.9945 | 5.9232 | 6.1949 |
| green | 200 | 0.4 | completed | total | 10 | 13.9865 | 13.9805 | 13.6540 | 14.4218 |
| green | 200 | 0.7 | completed | is | 10 | 0.6698 | 0.6407 | 0.4908 | 0.9452 |
| green | 200 | 0.7 | completed | oos | 10 | 6.0023 | 6.0009 | 5.8244 | 6.2356 |
| green | 200 | 0.7 | completed | total | 10 | 14.1633 | 14.0971 | 13.6176 | 14.6760 |
| green | 200 | 1.5 | completed | is | 10 | 1.0095 | 1.0345 | 0.6510 | 1.1966 |
| green | 200 | 1.5 | completed | oos | 10 | 5.9789 | 5.9914 | 5.8596 | 6.1196 |
| green | 200 | 1.5 | completed | total | 10 | 14.6678 | 14.6622 | 14.0281 | 15.0587 |
| green | 800 | 0 | completed | is | 10 | 0.4098 | 0.4043 | 0.3975 | 0.4476 |
| green | 800 | 0 | completed | oos | 10 | 6.0052 | 5.9985 | 5.7912 | 6.4181 |
| green | 800 | 0 | completed | total | 10 | 13.7842 | 13.6931 | 13.4130 | 14.8576 |
| green | 800 | 0.001 | completed | is | 10 | 0.4266 | 0.4109 | 0.4068 | 0.4838 |
| green | 800 | 0.001 | completed | oos | 10 | 6.0016 | 5.9466 | 5.8604 | 6.3735 |
| green | 800 | 0.001 | completed | total | 10 | 13.7169 | 13.7291 | 13.2260 | 14.3166 |
| green | 800 | 0.003 | completed | is | 10 | 0.4252 | 0.4207 | 0.3991 | 0.4817 |
| green | 800 | 0.003 | completed | oos | 10 | 6.0113 | 6.0203 | 5.7998 | 6.2067 |
| green | 800 | 0.003 | completed | total | 10 | 13.6712 | 13.6385 | 13.4031 | 14.0834 |
| green | 800 | 0.01 | completed | is | 10 | 0.4147 | 0.4097 | 0.3974 | 0.4535 |
| green | 800 | 0.01 | completed | oos | 10 | 5.9867 | 5.9886 | 5.7837 | 6.1520 |
| green | 800 | 0.01 | completed | total | 10 | 13.6136 | 13.6048 | 13.2100 | 13.9443 |
| green | 800 | 0.03 | completed | is | 10 | 0.4178 | 0.4123 | 0.3961 | 0.4518 |
| green | 800 | 0.03 | completed | oos | 10 | 5.9572 | 5.9041 | 5.8417 | 6.1262 |
| green | 800 | 0.03 | completed | total | 10 | 13.6468 | 13.6305 | 13.3593 | 14.0708 |
| green | 800 | 0.2 | completed | is | 10 | 0.9306 | 0.8486 | 0.7623 | 1.2555 |
| green | 800 | 0.2 | completed | oos | 10 | 5.9784 | 5.9602 | 5.8842 | 6.2168 |
| green | 800 | 0.2 | completed | total | 10 | 14.2943 | 14.2337 | 13.7895 | 15.0148 |
| green | 800 | 0.4 | completed | is | 10 | 1.4252 | 1.4872 | 1.1454 | 1.6071 |
| green | 800 | 0.4 | completed | oos | 10 | 6.0068 | 5.9877 | 5.7636 | 6.1912 |
| green | 800 | 0.4 | completed | total | 10 | 14.8204 | 14.8138 | 14.4898 | 15.2284 |
| green | 800 | 0.7 | completed | is | 10 | 1.9834 | 1.9083 | 1.6089 | 2.7252 |
| green | 800 | 0.7 | completed | oos | 10 | 5.9646 | 5.9476 | 5.7958 | 6.2410 |
| green | 800 | 0.7 | completed | total | 10 | 15.5523 | 15.3732 | 15.1674 | 16.6614 |
| green | 800 | 1.5 | completed | is | 10 | 3.3078 | 3.2075 | 2.5044 | 4.3716 |
| green | 800 | 1.5 | completed | oos | 10 | 5.9569 | 5.9687 | 5.8482 | 6.0858 |
| green | 800 | 1.5 | completed | total | 10 | 17.2308 | 17.4355 | 16.0522 | 18.2789 |
| scs | 50 | 0 | completed | is | 10 | 0.0579 | 0.0564 | 0.0439 | 0.0718 |
| scs | 50 | 0 | completed | oos | 10 | 6.0167 | 5.9904 | 5.8680 | 6.3091 |
| scs | 50 | 0 | completed | total | 10 | 13.2086 | 13.1552 | 13.0132 | 13.5851 |
| scs | 50 | 0.001 | completed | is | 10 | 0.1169 | 0.1169 | 0.0984 | 0.1405 |
| scs | 50 | 0.001 | completed | oos | 10 | 6.0654 | 6.0363 | 5.8945 | 6.2958 |
| scs | 50 | 0.001 | completed | total | 10 | 13.3001 | 13.2640 | 13.0692 | 13.7706 |
| scs | 50 | 0.005 | completed | is | 10 | 0.1127 | 0.1085 | 0.0740 | 0.1780 |
| scs | 50 | 0.005 | completed | oos | 10 | 6.0267 | 5.9827 | 5.8919 | 6.2325 |
| scs | 50 | 0.005 | completed | total | 10 | 13.3933 | 13.4999 | 12.8742 | 13.7267 |
| scs | 50 | 0.01 | completed | is | 10 | 0.1110 | 0.1107 | 0.0863 | 0.1549 |
| scs | 50 | 0.01 | completed | oos | 10 | 6.1007 | 6.0707 | 5.8571 | 6.4354 |
| scs | 50 | 0.01 | completed | total | 10 | 13.4482 | 13.3972 | 12.9081 | 13.9438 |
| scs | 50 | 0.02 | completed | is | 10 | 0.1033 | 0.1058 | 0.0745 | 0.1171 |
| scs | 50 | 0.02 | completed | oos | 10 | 6.3884 | 6.3340 | 6.2042 | 7.0108 |
| scs | 50 | 0.02 | completed | total | 10 | 13.7125 | 13.4500 | 13.3214 | 15.6021 |
| scs | 50 | 0.05 | completed | is | 10 | 0.1323 | 0.1230 | 0.1048 | 0.1842 |
| scs | 50 | 0.05 | completed | oos | 10 | 6.0279 | 6.0177 | 5.8948 | 6.1850 |
| scs | 50 | 0.05 | completed | total | 10 | 13.2624 | 13.2336 | 13.0130 | 13.6628 |
| scs | 50 | 0.1 | completed | is | 10 | 0.1617 | 0.1622 | 0.0863 | 0.2072 |
| scs | 50 | 0.1 | completed | oos | 10 | 6.0490 | 5.9959 | 5.8571 | 6.4392 |
| scs | 50 | 0.1 | completed | total | 10 | 13.4239 | 13.3923 | 12.8755 | 14.3852 |
| scs | 50 | 0.15 | completed | is | 10 | 0.2379 | 0.2176 | 0.1777 | 0.3586 |
| scs | 50 | 0.15 | completed | oos | 10 | 6.1141 | 6.0893 | 5.9607 | 6.4509 |
| scs | 50 | 0.15 | completed | total | 10 | 13.4398 | 13.4373 | 13.1129 | 13.9062 |
| scs | 50 | 0.2 | completed | is | 10 | 0.2752 | 0.2913 | 0.1800 | 0.3558 |
| scs | 50 | 0.2 | completed | oos | 10 | 5.9770 | 5.9637 | 5.9141 | 6.1317 |
| scs | 50 | 0.2 | completed | total | 10 | 13.3496 | 13.3436 | 13.0297 | 13.5921 |
| scs | 50 | 0.3 | completed | is | 10 | 0.3222 | 0.3130 | 0.1982 | 0.4173 |
| scs | 50 | 0.3 | completed | oos | 10 | 5.9893 | 6.0094 | 5.8351 | 6.1097 |
| scs | 50 | 0.3 | completed | total | 10 | 13.4921 | 13.5758 | 13.0410 | 13.9121 |
| scs | 50 | 0.4 | completed | is | 10 | 0.3717 | 0.3839 | 0.1297 | 0.7777 |
| scs | 50 | 0.4 | completed | oos | 10 | 6.1536 | 6.1082 | 5.9779 | 6.5323 |
| scs | 50 | 0.4 | completed | total | 10 | 13.7313 | 13.6892 | 13.0303 | 15.1797 |
| scs | 50 | 0.45 | completed | is | 10 | 0.3965 | 0.4241 | 0.1767 | 0.6499 |
| scs | 50 | 0.45 | completed | oos | 10 | 5.9979 | 6.0336 | 5.8415 | 6.1587 |
| scs | 50 | 0.45 | completed | total | 10 | 13.6100 | 13.5441 | 13.0560 | 14.1297 |
| scs | 50 | 0.5 | completed | is | 10 | 0.4258 | 0.4351 | 0.2986 | 0.5081 |
| scs | 50 | 0.5 | completed | oos | 10 | 6.0036 | 6.0395 | 5.8816 | 6.1478 |
| scs | 50 | 0.5 | completed | total | 10 | 13.6139 | 13.5521 | 13.3627 | 14.1737 |
| scs | 50 | 0.6 | completed | is | 10 | 0.4674 | 0.4559 | 0.2156 | 0.8169 |
| scs | 50 | 0.6 | completed | oos | 10 | 6.0131 | 5.9567 | 5.8653 | 6.4474 |
| scs | 50 | 0.6 | completed | total | 10 | 13.5843 | 13.6158 | 13.0997 | 14.0776 |
| scs | 50 | 0.75 | completed | is | 10 | 0.3800 | 0.3528 | 0.2577 | 0.5839 |
| scs | 50 | 0.75 | completed | oos | 10 | 6.0958 | 6.0994 | 5.8714 | 6.3934 |
| scs | 50 | 0.75 | completed | total | 10 | 13.5779 | 13.6162 | 13.1654 | 14.0613 |
| scs | 50 | 0.8 | completed | is | 10 | 0.3265 | 0.3250 | 0.1724 | 0.5077 |
| scs | 50 | 0.8 | completed | oos | 10 | 5.9703 | 5.9871 | 5.7795 | 6.1299 |
| scs | 50 | 0.8 | completed | total | 10 | 13.4482 | 13.4588 | 13.0465 | 13.8823 |
| scs | 50 | 1 | completed | is | 10 | 0.3282 | 0.3282 | 0.1897 | 0.5028 |
| scs | 50 | 1 | completed | oos | 10 | 5.9114 | 5.9015 | 5.7774 | 5.9938 |
| scs | 50 | 1 | completed | total | 10 | 13.3019 | 13.2558 | 12.9858 | 13.7620 |
| scs | 50 | 1.5 | completed | is | 10 | 0.2841 | 0.3177 | 0.1677 | 0.4018 |
| scs | 50 | 1.5 | completed | oos | 10 | 5.9376 | 5.9868 | 5.7602 | 6.0549 |
| scs | 50 | 1.5 | completed | total | 10 | 13.4526 | 13.4277 | 13.0214 | 13.8520 |
| scs | 200 | 0 | completed | is | 10 | 0.1591 | 0.1473 | 0.1347 | 0.2248 |
| scs | 200 | 0 | completed | oos | 10 | 6.0402 | 5.9992 | 5.8797 | 6.2632 |
| scs | 200 | 0 | completed | total | 10 | 13.3352 | 13.3484 | 12.9137 | 13.8456 |
| scs | 200 | 0.001 | completed | is | 10 | 0.2794 | 0.2678 | 0.2458 | 0.4341 |
| scs | 200 | 0.001 | completed | oos | 10 | 5.9791 | 5.9858 | 5.8774 | 6.0500 |
| scs | 200 | 0.001 | completed | total | 10 | 13.2823 | 13.2961 | 13.0848 | 13.4631 |
| scs | 200 | 0.005 | completed | is | 10 | 0.2724 | 0.2693 | 0.2584 | 0.2881 |
| scs | 200 | 0.005 | completed | oos | 10 | 6.0703 | 6.0391 | 5.8058 | 6.3031 |
| scs | 200 | 0.005 | completed | total | 10 | 13.5048 | 13.3521 | 13.1271 | 14.5132 |
| scs | 200 | 0.01 | completed | is | 10 | 0.2711 | 0.2576 | 0.2450 | 0.3872 |
| scs | 200 | 0.01 | completed | oos | 10 | 5.9924 | 5.9791 | 5.8276 | 6.2228 |
| scs | 200 | 0.01 | completed | total | 10 | 13.4732 | 13.4526 | 13.2033 | 13.8863 |
| scs | 200 | 0.02 | completed | is | 10 | 0.2621 | 0.2580 | 0.2456 | 0.2929 |
| scs | 200 | 0.02 | completed | oos | 10 | 5.9435 | 5.9140 | 5.7992 | 6.2799 |
| scs | 200 | 0.02 | completed | total | 10 | 13.3929 | 13.3585 | 13.1201 | 13.7716 |
| scs | 200 | 0.05 | completed | is | 10 | 0.2782 | 0.2789 | 0.2457 | 0.3155 |
| scs | 200 | 0.05 | completed | oos | 10 | 6.0244 | 5.9702 | 5.8967 | 6.3548 |
| scs | 200 | 0.05 | completed | total | 10 | 13.5159 | 13.5060 | 13.1379 | 13.9262 |
| scs | 200 | 0.1 | completed | is | 10 | 0.2628 | 0.2581 | 0.2560 | 0.2811 |
| scs | 200 | 0.1 | completed | oos | 10 | 5.9514 | 5.9571 | 5.8357 | 6.0477 |
| scs | 200 | 0.1 | completed | total | 10 | 13.4302 | 13.3684 | 13.0339 | 14.0059 |
| scs | 200 | 0.15 | completed | is | 10 | 0.3229 | 0.3294 | 0.2456 | 0.3922 |
| scs | 200 | 0.15 | completed | oos | 10 | 5.9265 | 5.9210 | 5.8049 | 6.0328 |
| scs | 200 | 0.15 | completed | total | 10 | 13.4892 | 13.3805 | 13.1103 | 14.1150 |
| scs | 200 | 0.2 | completed | is | 10 | 0.3555 | 0.3842 | 0.2673 | 0.4127 |
| scs | 200 | 0.2 | completed | oos | 10 | 6.0160 | 6.0037 | 5.9347 | 6.1363 |
| scs | 200 | 0.2 | completed | total | 10 | 13.4721 | 13.4077 | 13.2695 | 14.0118 |
| scs | 200 | 0.3 | completed | is | 10 | 0.4453 | 0.4505 | 0.3670 | 0.5226 |
| scs | 200 | 0.3 | completed | oos | 10 | 6.0228 | 6.0283 | 5.8172 | 6.3410 |
| scs | 200 | 0.3 | completed | total | 10 | 13.7006 | 13.7517 | 13.3481 | 13.9590 |
| scs | 200 | 0.4 | completed | is | 10 | 0.5060 | 0.5089 | 0.4898 | 0.5192 |
| scs | 200 | 0.4 | completed | oos | 10 | 5.9810 | 5.9709 | 5.8380 | 6.1296 |
| scs | 200 | 0.4 | completed | total | 10 | 13.6693 | 13.7222 | 13.2930 | 14.1234 |
| scs | 200 | 0.45 | completed | is | 10 | 0.5565 | 0.5179 | 0.4901 | 0.6537 |
| scs | 200 | 0.45 | completed | oos | 10 | 6.0801 | 6.0032 | 5.9086 | 6.4486 |
| scs | 200 | 0.45 | completed | total | 10 | 13.8477 | 13.7677 | 13.5291 | 14.4162 |
| scs | 200 | 0.5 | completed | is | 10 | 0.5626 | 0.5565 | 0.4774 | 0.6709 |
| scs | 200 | 0.5 | completed | oos | 10 | 5.9989 | 5.9907 | 5.8894 | 6.1272 |
| scs | 200 | 0.5 | completed | total | 10 | 13.8173 | 13.8008 | 13.5500 | 14.1256 |
| scs | 200 | 0.6 | completed | is | 10 | 0.6436 | 0.6207 | 0.5000 | 0.7669 |
| scs | 200 | 0.6 | completed | oos | 10 | 6.0240 | 6.0050 | 5.8819 | 6.1891 |
| scs | 200 | 0.6 | completed | total | 10 | 13.9378 | 13.9410 | 13.5872 | 14.2298 |
| scs | 200 | 0.75 | completed | is | 10 | 0.7444 | 0.6863 | 0.5983 | 1.0220 |
| scs | 200 | 0.75 | completed | oos | 10 | 5.9906 | 5.9760 | 5.8550 | 6.2083 |
| scs | 200 | 0.75 | completed | total | 10 | 13.9484 | 13.9368 | 13.5342 | 14.4630 |
| scs | 200 | 0.8 | completed | is | 10 | 0.7635 | 0.7533 | 0.5991 | 0.9772 |
| scs | 200 | 0.8 | completed | oos | 10 | 6.0323 | 5.9848 | 5.8515 | 6.5026 |
| scs | 200 | 0.8 | completed | total | 10 | 13.8976 | 13.8960 | 13.3244 | 14.3986 |
| scs | 200 | 1 | completed | is | 10 | 0.9119 | 0.8735 | 0.6577 | 1.1454 |
| scs | 200 | 1 | completed | oos | 10 | 5.9544 | 5.9332 | 5.8439 | 6.0955 |
| scs | 200 | 1 | completed | total | 10 | 14.2467 | 14.1776 | 13.7587 | 14.8273 |
| scs | 200 | 1.5 | completed | is | 10 | 1.1592 | 1.1139 | 0.9074 | 1.5411 |
| scs | 200 | 1.5 | completed | oos | 10 | 5.9989 | 6.0040 | 5.8905 | 6.2160 |
| scs | 200 | 1.5 | completed | total | 10 | 14.4359 | 14.3449 | 14.0259 | 15.2967 |
| scs | 800 | 0 | completed | is | 10 | 0.5096 | 0.5115 | 0.4679 | 0.5482 |
| scs | 800 | 0 | completed | oos | 10 | 5.9476 | 5.9320 | 5.8209 | 6.1105 |
| scs | 800 | 0 | completed | total | 10 | 13.5811 | 13.5874 | 13.1782 | 13.7826 |
| scs | 800 | 0.001 | completed | is | 10 | 0.9805 | 0.9742 | 0.9630 | 1.0411 |
| scs | 800 | 0.001 | completed | oos | 10 | 5.9692 | 5.9635 | 5.8560 | 6.0573 |
| scs | 800 | 0.001 | completed | total | 10 | 14.2679 | 14.2779 | 13.9354 | 14.5529 |
| scs | 800 | 0.005 | completed | is | 10 | 1.0087 | 1.0093 | 0.9725 | 1.0502 |
| scs | 800 | 0.005 | completed | oos | 10 | 5.9825 | 5.9779 | 5.8440 | 6.2610 |
| scs | 800 | 0.005 | completed | total | 10 | 14.3232 | 14.3345 | 14.0308 | 14.6640 |
| scs | 800 | 0.01 | completed | is | 10 | 0.9897 | 0.9839 | 0.9626 | 1.0327 |
| scs | 800 | 0.01 | completed | oos | 10 | 6.0754 | 6.0397 | 5.8426 | 6.4510 |
| scs | 800 | 0.01 | completed | total | 10 | 14.3001 | 14.3058 | 13.8611 | 14.6178 |
| scs | 800 | 0.02 | completed | is | 10 | 0.9685 | 0.9647 | 0.9520 | 0.9952 |
| scs | 800 | 0.02 | completed | oos | 10 | 5.9165 | 5.9014 | 5.8228 | 6.0398 |
| scs | 800 | 0.02 | completed | total | 10 | 14.0128 | 13.9512 | 13.6658 | 14.8039 |
| scs | 800 | 0.05 | completed | is | 10 | 1.0059 | 0.9920 | 0.9619 | 1.1014 |
| scs | 800 | 0.05 | completed | oos | 10 | 5.9529 | 5.9484 | 5.8818 | 6.0556 |
| scs | 800 | 0.05 | completed | total | 10 | 14.2078 | 14.1546 | 13.8302 | 14.8402 |
| scs | 800 | 0.1 | completed | is | 10 | 1.0121 | 1.0073 | 0.9735 | 1.0636 |
| scs | 800 | 0.1 | completed | oos | 10 | 6.1067 | 6.0820 | 5.9901 | 6.3590 |
| scs | 800 | 0.1 | completed | total | 10 | 14.4224 | 14.3578 | 14.1886 | 14.8888 |
| scs | 800 | 0.15 | completed | is | 10 | 0.9724 | 0.9698 | 0.9520 | 0.9936 |
| scs | 800 | 0.15 | completed | oos | 10 | 5.9869 | 5.9664 | 5.8675 | 6.1199 |
| scs | 800 | 0.15 | completed | total | 10 | 14.1746 | 14.1829 | 13.9120 | 14.4584 |
| scs | 800 | 0.2 | completed | is | 10 | 1.2806 | 1.4562 | 0.9710 | 1.4989 |
| scs | 800 | 0.2 | completed | oos | 10 | 5.9185 | 5.9119 | 5.7576 | 6.0844 |
| scs | 800 | 0.2 | completed | total | 10 | 14.3942 | 14.4329 | 13.7185 | 15.0545 |
| scs | 800 | 0.3 | completed | is | 10 | 1.3912 | 1.4871 | 0.5099 | 1.6786 |
| scs | 800 | 0.3 | completed | oos | 10 | 6.0262 | 6.0187 | 5.8611 | 6.2676 |
| scs | 800 | 0.3 | completed | total | 10 | 14.5840 | 14.6544 | 13.5936 | 15.0028 |
| scs | 800 | 0.4 | completed | is | 10 | 1.4344 | 1.4701 | 0.9531 | 2.1028 |
| scs | 800 | 0.4 | completed | oos | 10 | 6.0623 | 6.0830 | 5.8390 | 6.1885 |
| scs | 800 | 0.4 | completed | total | 10 | 14.8477 | 14.8900 | 14.0099 | 15.6543 |
| scs | 800 | 0.45 | completed | is | 10 | 1.6005 | 1.4667 | 1.4274 | 1.9624 |
| scs | 800 | 0.45 | completed | oos | 10 | 5.9214 | 5.9489 | 5.8369 | 6.0040 |
| scs | 800 | 0.45 | completed | total | 10 | 14.8030 | 14.8220 | 14.3583 | 15.3521 |
| scs | 800 | 0.5 | completed | is | 10 | 1.4824 | 1.4675 | 1.0044 | 1.9916 |
| scs | 800 | 0.5 | completed | oos | 10 | 5.9853 | 5.9385 | 5.8818 | 6.1979 |
| scs | 800 | 0.5 | completed | total | 10 | 14.6525 | 14.6258 | 13.9647 | 15.3863 |
| scs | 800 | 0.6 | completed | is | 10 | 1.7082 | 1.7171 | 1.4066 | 2.0282 |
| scs | 800 | 0.6 | completed | oos | 10 | 6.0327 | 6.0142 | 5.8205 | 6.3568 |
| scs | 800 | 0.6 | completed | total | 10 | 15.2587 | 15.1477 | 14.2625 | 16.5602 |
| scs | 800 | 0.75 | completed | is | 10 | 1.7471 | 1.6875 | 1.4579 | 2.3719 |
| scs | 800 | 0.75 | completed | oos | 10 | 5.9927 | 5.9673 | 5.7229 | 6.2551 |
| scs | 800 | 0.75 | completed | total | 10 | 15.0604 | 15.1139 | 14.5965 | 15.5311 |
| scs | 800 | 0.8 | completed | is | 10 | 1.7989 | 1.9182 | 1.4374 | 2.0649 |
| scs | 800 | 0.8 | completed | oos | 10 | 5.9535 | 5.9353 | 5.8783 | 6.1110 |
| scs | 800 | 0.8 | completed | total | 10 | 15.0461 | 15.1089 | 14.5252 | 15.3955 |
| scs | 800 | 1 | completed | is | 10 | 2.0180 | 1.9603 | 1.8883 | 2.4129 |
| scs | 800 | 1 | completed | oos | 10 | 5.9777 | 5.9667 | 5.8270 | 6.1380 |
| scs | 800 | 1 | completed | total | 10 | 15.2920 | 15.2303 | 14.9489 | 15.8161 |
| scs | 800 | 1.5 | completed | is | 10 | 2.6372 | 2.3763 | 1.8942 | 3.8219 |
| scs | 800 | 1.5 | completed | oos | 10 | 5.8173 | 5.8154 | 5.6838 | 6.0107 |
| scs | 800 | 1.5 | completed | total | 10 | 15.7232 | 15.5184 | 14.7840 | 16.8514 |

## Persisted historical training / reference timings

These are old-run times, never current cache-loading times. Historical hardware/concurrency is only
known where recorded in the frozen source; otherwise unavailable. Missing times require a rerun.

| Method / scope | N | Radius / LR | n | Mean s | Median s | Min s | Max s |
|---|---:|---|---:|---:|---:|---:|---:|
| Green nominal training | 50 | LR=0.005 | 10 | 125.9823 | 126.4167 | 117.5035 | 130.8335 |
| Green nominal train evaluation | 50 | LR=0.005 | 10 | 0.0196 | 0.0193 | 0.0124 | 0.0282 |
| Green nominal training | 50 | LR=0.01 | 10 | 127.9994 | 126.8636 | 125.6904 | 130.7324 |
| Green nominal train evaluation | 50 | LR=0.01 | 10 | 0.0180 | 0.0179 | 0.0139 | 0.0235 |
| Green nominal training | 50 | LR=0.05 | 10 | 97.4773 | 85.4843 | 82.6408 | 129.8009 |
| Green nominal train evaluation | 50 | LR=0.05 | 10 | 0.0111 | 0.0105 | 0.0052 | 0.0184 |
| Green CCG total (excludes shared nominal) | 50 | 0.0 | 10 | 4.2419 | 5.0203 | 0.8732 | 5.1111 |
| Green CCG total (excludes shared nominal) | 50 | 0.001 | 10 | 3.4691 | 2.4290 | 2.2714 | 5.2151 |
| Green CCG total (excludes shared nominal) | 50 | 0.003 | 10 | 1.8506 | 1.6445 | 1.5362 | 2.4584 |
| Green CCG total (excludes shared nominal) | 50 | 0.01 | 10 | 1.6978 | 1.6901 | 1.6088 | 1.8019 |
| Green CCG total (excludes shared nominal) | 50 | 0.03 | 10 | 35.4951 | 1.8900 | 1.7500 | 229.0941 |
| Green CCG total (excludes shared nominal) | 50 | 0.2 | 10 | 924.4195 | 945.7675 | 426.8013 | 1325.0765 |
| Green CCG total (excludes shared nominal) | 50 | 0.4 | 10 | 1901.6072 | 2010.8123 | 837.8168 | 3370.8289 |
| Green CCG total (excludes shared nominal) | 50 | 0.7 | 10 | 2652.5435 | 2201.6729 | 1342.6369 | 5096.2476 |
| Green CCG total (excludes shared nominal) | 50 | 1.5 | 10 | 3571.2893 | 3839.9000 | 1084.0615 | 5106.1436 |
| Green initialization | 50 | all | 10 | 0.1960 | 0.1032 | 0.0718 | 0.5469 |
| Green lr_validation | 50 | all | 10 | 0.0467 | 0.0403 | 0.0331 | 0.0842 |
| Green radius_validation | 50 | all | 10 | 0.3421 | 0.3632 | 0.2448 | 0.4174 |
| Green nominal training | 200 | LR=0.005 | 10 | 167.0817 | 167.2480 | 159.6194 | 170.5565 |
| Green nominal train evaluation | 200 | LR=0.005 | 10 | 0.0238 | 0.0240 | 0.0189 | 0.0302 |
| Green nominal training | 200 | LR=0.01 | 10 | 169.2397 | 170.1210 | 166.1811 | 170.8916 |
| Green nominal train evaluation | 200 | LR=0.01 | 10 | 0.0174 | 0.0175 | 0.0107 | 0.0260 |
| Green nominal training | 200 | LR=0.05 | 10 | 113.5206 | 100.1071 | 94.6022 | 170.4413 |
| Green nominal train evaluation | 200 | LR=0.05 | 10 | 0.0091 | 0.0076 | 0.0052 | 0.0200 |
| Green CCG total (excludes shared nominal) | 200 | 0.0 | 10 | 9.1920 | 9.2757 | 8.4377 | 9.6534 |
| Green CCG total (excludes shared nominal) | 200 | 0.001 | 10 | 7.8284 | 6.8383 | 6.6490 | 9.5676 |
| Green CCG total (excludes shared nominal) | 200 | 0.003 | 10 | 6.2648 | 6.1347 | 5.8568 | 6.7684 |
| Green CCG total (excludes shared nominal) | 200 | 0.01 | 10 | 5.9594 | 5.9486 | 5.6325 | 6.3740 |
| Green CCG total (excludes shared nominal) | 200 | 0.03 | 10 | 6.3853 | 6.3540 | 6.0560 | 6.7997 |
| Green CCG total (excludes shared nominal) | 200 | 0.2 | 10 | 403.8121 | 416.9384 | 42.8387 | 639.6838 |
| Green CCG total (excludes shared nominal) | 200 | 0.4 | 10 | 895.8398 | 950.6736 | 687.0851 | 1314.1612 |
| Green CCG total (excludes shared nominal) | 200 | 0.7 | 10 | 1694.1364 | 1570.3481 | 966.6649 | 2704.5429 |
| Green CCG total (excludes shared nominal) | 200 | 1.5 | 10 | 4841.7341 | 4483.1770 | 2449.9884 | 6557.4057 |
| Green initialization | 200 | all | 10 | 0.3391 | 0.2977 | 0.2094 | 0.5543 |
| Green lr_validation | 200 | all | 10 | 0.0497 | 0.0461 | 0.0407 | 0.0627 |
| Green radius_validation | 200 | all | 10 | 0.3100 | 0.3142 | 0.2702 | 0.3563 |
| Green nominal training | 800 | LR=0.005 | 10 | 215.7097 | 216.0481 | 213.4932 | 216.6354 |
| Green nominal train evaluation | 800 | LR=0.005 | 10 | 0.0213 | 0.0210 | 0.0122 | 0.0287 |
| Green nominal training | 800 | LR=0.01 | 10 | 216.4005 | 216.4408 | 215.5082 | 216.9845 |
| Green nominal train evaluation | 800 | LR=0.01 | 10 | 0.0186 | 0.0196 | 0.0132 | 0.0237 |
| Green nominal training | 800 | LR=0.05 | 10 | 145.4088 | 130.1045 | 120.6991 | 216.5773 |
| Green nominal train evaluation | 800 | LR=0.05 | 10 | 0.0171 | 0.0184 | 0.0114 | 0.0225 |
| Green CCG total (excludes shared nominal) | 800 | 0.0 | 10 | 25.7915 | 25.8892 | 24.6907 | 26.2975 |
| Green CCG total (excludes shared nominal) | 800 | 0.001 | 10 | 24.5678 | 23.7597 | 23.5146 | 26.7317 |
| Green CCG total (excludes shared nominal) | 800 | 0.003 | 10 | 23.5737 | 23.4974 | 22.9785 | 24.2687 |
| Green CCG total (excludes shared nominal) | 800 | 0.01 | 10 | 23.7365 | 23.7075 | 22.8396 | 24.4223 |
| Green CCG total (excludes shared nominal) | 800 | 0.03 | 10 | 25.4105 | 25.0876 | 23.9609 | 26.7566 |
| Green CCG total (excludes shared nominal) | 800 | 0.2 | 10 | 584.3224 | 454.9959 | 417.5943 | 943.5109 |
| Green CCG total (excludes shared nominal) | 800 | 0.4 | 10 | 1361.1641 | 1505.5292 | 946.9887 | 1711.7463 |
| Green CCG total (excludes shared nominal) | 800 | 0.7 | 10 | 2774.9694 | 2504.3207 | 1626.3475 | 4868.8505 |
| Green CCG total (excludes shared nominal) | 800 | 1.5 | 10 | 7990.3630 | 6882.7365 | 3895.8384 | 18468.4064 |
| Green initialization | 800 | all | 10 | 1.0231 | 0.9813 | 0.9518 | 1.2781 |
| Green lr_validation | 800 | all | 10 | 0.0646 | 0.0602 | 0.0525 | 0.0841 |
| Green radius_validation | 800 | all | 10 | 0.3949 | 0.3932 | 0.3395 | 0.4367 |
| SCS nominal/full CCG (not certified) | 50 | all | 0 | unavailable | unavailable | unavailable | unavailable |
| SRO-FA historical partition | 50 | 0.001 | 10 | 0.0017 | 0.0017 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.001 | 10 | 0.0077 | 0.0073 | 0.0066 | 0.0090 |
| SRO-FA historical OOS | 50 | 0.001 | 10 | 0.0139 | 0.0140 | 0.0135 | 0.0143 |
| SRO-FA historical total | 50 | 0.001 | 10 | 0.0899 | 0.0820 | 0.0700 | 0.1194 |
| SRO-FA historical partition | 50 | 0.005 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0019 |
| SRO-FA historical solver | 50 | 0.005 | 10 | 0.0077 | 0.0075 | 0.0067 | 0.0091 |
| SRO-FA historical OOS | 50 | 0.005 | 10 | 0.0138 | 0.0139 | 0.0130 | 0.0148 |
| SRO-FA historical total | 50 | 0.005 | 10 | 0.0823 | 0.0813 | 0.0695 | 0.1027 |
| SRO-FA historical partition | 50 | 0.01 | 10 | 0.0019 | 0.0018 | 0.0017 | 0.0028 |
| SRO-FA historical solver | 50 | 0.01 | 10 | 0.0085 | 0.0082 | 0.0077 | 0.0099 |
| SRO-FA historical OOS | 50 | 0.01 | 10 | 0.0134 | 0.0132 | 0.0129 | 0.0141 |
| SRO-FA historical total | 50 | 0.01 | 10 | 0.0817 | 0.0797 | 0.0692 | 0.1047 |
| SRO-FA historical partition | 50 | 0.02 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.02 | 10 | 0.0092 | 0.0093 | 0.0081 | 0.0112 |
| SRO-FA historical OOS | 50 | 0.02 | 10 | 0.0135 | 0.0135 | 0.0132 | 0.0139 |
| SRO-FA historical total | 50 | 0.02 | 10 | 0.0843 | 0.0849 | 0.0694 | 0.0999 |
| SRO-FA historical partition | 50 | 0.05 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.05 | 10 | 0.0101 | 0.0098 | 0.0086 | 0.0133 |
| SRO-FA historical OOS | 50 | 0.05 | 10 | 0.0135 | 0.0133 | 0.0127 | 0.0143 |
| SRO-FA historical total | 50 | 0.05 | 10 | 0.0921 | 0.0916 | 0.0791 | 0.1127 |
| SRO-FA historical partition | 50 | 0.1 | 10 | 0.0022 | 0.0018 | 0.0017 | 0.0062 |
| SRO-FA historical solver | 50 | 0.1 | 10 | 0.0124 | 0.0130 | 0.0101 | 0.0143 |
| SRO-FA historical OOS | 50 | 0.1 | 10 | 0.0135 | 0.0135 | 0.0126 | 0.0153 |
| SRO-FA historical total | 50 | 0.1 | 10 | 0.1017 | 0.0984 | 0.0892 | 0.1249 |
| SRO-FA historical partition | 50 | 0.15 | 10 | 0.0017 | 0.0018 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.15 | 10 | 0.0151 | 0.0152 | 0.0126 | 0.0173 |
| SRO-FA historical OOS | 50 | 0.15 | 10 | 0.0135 | 0.0136 | 0.0128 | 0.0143 |
| SRO-FA historical total | 50 | 0.15 | 10 | 0.1133 | 0.1119 | 0.0891 | 0.1395 |
| SRO-FA historical partition | 50 | 0.2 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0019 |
| SRO-FA historical solver | 50 | 0.2 | 10 | 0.0191 | 0.0191 | 0.0160 | 0.0237 |
| SRO-FA historical OOS | 50 | 0.2 | 10 | 0.0133 | 0.0130 | 0.0129 | 0.0143 |
| SRO-FA historical total | 50 | 0.2 | 10 | 0.1564 | 0.1196 | 0.0995 | 0.5159 |
| SRO-FA historical partition | 50 | 0.3 | 10 | 0.0017 | 0.0017 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.3 | 10 | 0.0288 | 0.0296 | 0.0180 | 0.0353 |
| SRO-FA historical OOS | 50 | 0.3 | 10 | 0.0128 | 0.0128 | 0.0122 | 0.0134 |
| SRO-FA historical total | 50 | 0.3 | 10 | 0.1444 | 0.1430 | 0.1399 | 0.1516 |
| SRO-FA historical partition | 50 | 0.4 | 10 | 0.0017 | 0.0018 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.4 | 10 | 0.0400 | 0.0389 | 0.0321 | 0.0510 |
| SRO-FA historical OOS | 50 | 0.4 | 10 | 0.0133 | 0.0131 | 0.0128 | 0.0147 |
| SRO-FA historical total | 50 | 0.4 | 10 | 0.1729 | 0.1725 | 0.1520 | 0.2090 |
| SRO-FA historical partition | 50 | 0.45 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.45 | 10 | 0.0467 | 0.0449 | 0.0315 | 0.0605 |
| SRO-FA historical OOS | 50 | 0.45 | 10 | 0.0137 | 0.0138 | 0.0129 | 0.0144 |
| SRO-FA historical total | 50 | 0.45 | 10 | 0.1955 | 0.2012 | 0.1514 | 0.2288 |
| SRO-FA historical partition | 50 | 0.5 | 10 | 0.0019 | 0.0018 | 0.0017 | 0.0027 |
| SRO-FA historical solver | 50 | 0.5 | 10 | 0.0494 | 0.0488 | 0.0373 | 0.0627 |
| SRO-FA historical OOS | 50 | 0.5 | 10 | 0.0137 | 0.0136 | 0.0129 | 0.0151 |
| SRO-FA historical total | 50 | 0.5 | 10 | 0.1956 | 0.1956 | 0.1506 | 0.2223 |
| SRO-FA historical partition | 50 | 0.6 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.6 | 10 | 0.0643 | 0.0612 | 0.0485 | 0.0835 |
| SRO-FA historical OOS | 50 | 0.6 | 10 | 0.0135 | 0.0134 | 0.0126 | 0.0147 |
| SRO-FA historical total | 50 | 0.6 | 10 | 0.2277 | 0.2244 | 0.1925 | 0.2723 |
| SRO-FA historical partition | 50 | 0.75 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 0.75 | 10 | 0.0879 | 0.0846 | 0.0716 | 0.1098 |
| SRO-FA historical OOS | 50 | 0.75 | 10 | 0.0130 | 0.0131 | 0.0121 | 0.0139 |
| SRO-FA historical total | 50 | 0.75 | 10 | 0.2753 | 0.2683 | 0.2328 | 0.3385 |
| SRO-FA historical partition | 50 | 0.8 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0019 |
| SRO-FA historical solver | 50 | 0.8 | 10 | 0.1010 | 0.0938 | 0.0664 | 0.1387 |
| SRO-FA historical OOS | 50 | 0.8 | 10 | 0.0132 | 0.0133 | 0.0123 | 0.0141 |
| SRO-FA historical total | 50 | 0.8 | 10 | 0.3025 | 0.3181 | 0.2310 | 0.3706 |
| SRO-FA historical partition | 50 | 1.0 | 10 | 0.0018 | 0.0018 | 0.0017 | 0.0018 |
| SRO-FA historical solver | 50 | 1.0 | 10 | 0.1563 | 0.1623 | 0.0940 | 0.2123 |
| SRO-FA historical OOS | 50 | 1.0 | 10 | 0.0131 | 0.0130 | 0.0120 | 0.0144 |
| SRO-FA historical total | 50 | 1.0 | 10 | 0.3999 | 0.3988 | 0.3358 | 0.4961 |
| SRO-FA historical partition | 50 | 1.5 | 10 | 0.0016 | 0.0016 | 0.0016 | 0.0018 |
| SRO-FA historical solver | 50 | 1.5 | 10 | 0.3804 | 0.3614 | 0.3328 | 0.5218 |
| SRO-FA historical OOS | 50 | 1.5 | 10 | 0.0121 | 0.0119 | 0.0110 | 0.0133 |
| SRO-FA historical total | 50 | 1.5 | 10 | 0.7322 | 0.7024 | 0.6637 | 0.8932 |
| SCS nominal/full CCG (not certified) | 200 | all | 0 | unavailable | unavailable | unavailable | unavailable |
| SRO-FA historical partition | 200 | 0.001 | 10 | 0.0098 | 0.0099 | 0.0097 | 0.0099 |
| SRO-FA historical solver | 200 | 0.001 | 10 | 0.0466 | 0.0446 | 0.0416 | 0.0607 |
| SRO-FA historical OOS | 200 | 0.001 | 10 | 0.0409 | 0.0407 | 0.0380 | 0.0476 |
| SRO-FA historical total | 200 | 0.001 | 10 | 0.2954 | 0.2879 | 0.2572 | 0.3446 |
| SRO-FA historical partition | 200 | 0.005 | 10 | 0.0099 | 0.0099 | 0.0098 | 0.0100 |
| SRO-FA historical solver | 200 | 0.005 | 10 | 0.0476 | 0.0477 | 0.0402 | 0.0520 |
| SRO-FA historical OOS | 200 | 0.005 | 10 | 0.0412 | 0.0413 | 0.0389 | 0.0423 |
| SRO-FA historical total | 200 | 0.005 | 10 | 0.2890 | 0.2847 | 0.2673 | 0.3123 |
| SRO-FA historical partition | 200 | 0.01 | 10 | 0.0099 | 0.0099 | 0.0098 | 0.0100 |
| SRO-FA historical solver | 200 | 0.01 | 10 | 0.0492 | 0.0489 | 0.0445 | 0.0538 |
| SRO-FA historical OOS | 200 | 0.01 | 10 | 0.0418 | 0.0412 | 0.0402 | 0.0475 |
| SRO-FA historical total | 200 | 0.01 | 10 | 0.2936 | 0.2933 | 0.2786 | 0.3043 |
| SRO-FA historical partition | 200 | 0.02 | 10 | 0.0099 | 0.0099 | 0.0098 | 0.0100 |
| SRO-FA historical solver | 200 | 0.02 | 10 | 0.0522 | 0.0529 | 0.0473 | 0.0574 |
| SRO-FA historical OOS | 200 | 0.02 | 10 | 0.0421 | 0.0412 | 0.0406 | 0.0488 |
| SRO-FA historical total | 200 | 0.02 | 10 | 0.3124 | 0.3161 | 0.2783 | 0.3427 |
| SRO-FA historical partition | 200 | 0.05 | 10 | 0.0099 | 0.0099 | 0.0098 | 0.0104 |
| SRO-FA historical solver | 200 | 0.05 | 10 | 0.0619 | 0.0605 | 0.0548 | 0.0716 |
| SRO-FA historical OOS | 200 | 0.05 | 10 | 0.0418 | 0.0411 | 0.0403 | 0.0475 |
| SRO-FA historical total | 200 | 0.05 | 10 | 0.3302 | 0.3293 | 0.3089 | 0.3534 |
| SRO-FA historical partition | 200 | 0.1 | 10 | 0.0100 | 0.0099 | 0.0098 | 0.0105 |
| SRO-FA historical solver | 200 | 0.1 | 10 | 0.0864 | 0.0870 | 0.0757 | 0.1027 |
| SRO-FA historical OOS | 200 | 0.1 | 10 | 0.0428 | 0.0415 | 0.0408 | 0.0484 |
| SRO-FA historical total | 200 | 0.1 | 10 | 0.3883 | 0.3910 | 0.3700 | 0.4113 |
| SRO-FA historical partition | 200 | 0.15 | 10 | 0.0098 | 0.0099 | 0.0090 | 0.0100 |
| SRO-FA historical solver | 200 | 0.15 | 10 | 0.1118 | 0.1106 | 0.0859 | 0.1452 |
| SRO-FA historical OOS | 200 | 0.15 | 10 | 0.0414 | 0.0423 | 0.0375 | 0.0430 |
| SRO-FA historical total | 200 | 0.15 | 10 | 0.4424 | 0.4415 | 0.3652 | 0.5135 |
| SRO-FA historical partition | 200 | 0.2 | 10 | 0.0091 | 0.0091 | 0.0090 | 0.0091 |
| SRO-FA historical solver | 200 | 0.2 | 10 | 0.1270 | 0.1241 | 0.0965 | 0.1742 |
| SRO-FA historical OOS | 200 | 0.2 | 10 | 0.0380 | 0.0381 | 0.0370 | 0.0398 |
| SRO-FA historical total | 200 | 0.2 | 10 | 0.4597 | 0.4540 | 0.4064 | 0.5200 |
| SRO-FA historical partition | 200 | 0.3 | 10 | 0.0091 | 0.0091 | 0.0090 | 0.0091 |
| SRO-FA historical solver | 200 | 0.3 | 10 | 0.2104 | 0.2063 | 0.1537 | 0.2795 |
| SRO-FA historical OOS | 200 | 0.3 | 10 | 0.0388 | 0.0389 | 0.0369 | 0.0398 |
| SRO-FA historical total | 200 | 0.3 | 10 | 0.6071 | 0.6150 | 0.5259 | 0.6994 |
| SRO-FA historical partition | 200 | 0.4 | 10 | 0.0091 | 0.0091 | 0.0090 | 0.0091 |
| SRO-FA historical solver | 200 | 0.4 | 10 | 0.3527 | 0.3329 | 0.2798 | 0.4582 |
| SRO-FA historical OOS | 200 | 0.4 | 10 | 0.0366 | 0.0369 | 0.0347 | 0.0388 |
| SRO-FA historical total | 200 | 0.4 | 10 | 0.8168 | 0.8096 | 0.7179 | 0.9497 |
| SRO-FA historical partition | 200 | 0.45 | 10 | 0.0093 | 0.0091 | 0.0089 | 0.0107 |
| SRO-FA historical solver | 200 | 0.45 | 10 | 0.4842 | 0.5125 | 0.3572 | 0.5781 |
| SRO-FA historical OOS | 200 | 0.45 | 10 | 0.0402 | 0.0388 | 0.0385 | 0.0453 |
| SRO-FA historical total | 200 | 0.45 | 10 | 1.0148 | 1.0275 | 0.8928 | 1.1519 |
| SRO-FA historical partition | 200 | 0.5 | 10 | 0.0090 | 0.0090 | 0.0090 | 0.0091 |
| SRO-FA historical solver | 200 | 0.5 | 10 | 0.5731 | 0.5722 | 0.4349 | 0.6809 |
| SRO-FA historical OOS | 200 | 0.5 | 10 | 0.0377 | 0.0371 | 0.0368 | 0.0389 |
| SRO-FA historical total | 200 | 0.5 | 10 | 1.1327 | 1.1338 | 1.0002 | 1.2266 |
| SRO-FA historical partition | 200 | 0.6 | 10 | 0.0090 | 0.0090 | 0.0089 | 0.0092 |
| SRO-FA historical solver | 200 | 0.6 | 10 | 0.8741 | 0.9129 | 0.6229 | 1.1325 |
| SRO-FA historical OOS | 200 | 0.6 | 10 | 0.0383 | 0.0385 | 0.0363 | 0.0393 |
| SRO-FA historical total | 200 | 0.6 | 10 | 1.5355 | 1.5721 | 1.2671 | 1.7908 |
| SRO-FA historical partition | 200 | 0.75 | 10 | 0.0090 | 0.0090 | 0.0089 | 0.0092 |
| SRO-FA historical solver | 200 | 0.75 | 10 | 1.9354 | 1.9686 | 1.2913 | 2.5774 |
| SRO-FA historical OOS | 200 | 0.75 | 10 | 0.0362 | 0.0362 | 0.0346 | 0.0379 |
| SRO-FA historical total | 200 | 0.75 | 10 | 2.7510 | 2.7846 | 2.0747 | 3.3739 |
| SRO-FA historical partition | 200 | 0.8 | 10 | 0.0090 | 0.0090 | 0.0089 | 0.0093 |
| SRO-FA historical solver | 200 | 0.8 | 10 | 3.1420 | 3.0348 | 2.3523 | 4.0129 |
| SRO-FA historical OOS | 200 | 0.8 | 10 | 0.0368 | 0.0362 | 0.0346 | 0.0402 |
| SRO-FA historical total | 200 | 0.8 | 10 | 4.0263 | 3.8917 | 3.2458 | 4.9289 |
| SRO-FA historical partition | 200 | 1.0 | 10 | 0.0091 | 0.0091 | 0.0090 | 0.0091 |
| SRO-FA historical solver | 200 | 1.0 | 10 | 5.6946 | 5.6974 | 4.7167 | 6.6924 |
| SRO-FA historical OOS | 200 | 1.0 | 10 | 0.0367 | 0.0368 | 0.0346 | 0.0378 |
| SRO-FA historical total | 200 | 1.0 | 10 | 6.8907 | 6.9339 | 5.8348 | 7.9214 |
| SRO-FA historical partition | 200 | 1.5 | 10 | 0.0093 | 0.0093 | 0.0092 | 0.0095 |
| SRO-FA historical solver | 200 | 1.5 | 10 | 26.0723 | 26.9087 | 20.8442 | 29.8464 |
| SRO-FA historical OOS | 200 | 1.5 | 10 | 0.0381 | 0.0384 | 0.0368 | 0.0393 |
| SRO-FA historical total | 200 | 1.5 | 10 | 28.4670 | 29.3375 | 23.1155 | 32.3926 |
| SCS nominal/full CCG (not certified) | 800 | all | 0 | unavailable | unavailable | unavailable | unavailable |
| SRO-FA historical partition | 800 | 0.001 | 10 | 0.0859 | 0.0872 | 0.0807 | 0.0899 |
| SRO-FA historical solver | 800 | 0.001 | 10 | 0.3331 | 0.3245 | 0.2751 | 0.4389 |
| SRO-FA historical OOS | 800 | 0.001 | 10 | 0.1860 | 0.1851 | 0.1696 | 0.1996 |
| SRO-FA historical total | 800 | 0.001 | 10 | 1.2545 | 1.2678 | 1.1302 | 1.4310 |
| SRO-FA historical partition | 800 | 0.005 | 10 | 0.0817 | 0.0816 | 0.0806 | 0.0841 |
| SRO-FA historical solver | 800 | 0.005 | 10 | 0.3064 | 0.2916 | 0.2657 | 0.3884 |
| SRO-FA historical OOS | 800 | 0.005 | 10 | 0.1733 | 0.1733 | 0.1632 | 0.1820 |
| SRO-FA historical total | 800 | 0.005 | 10 | 1.1958 | 1.1911 | 1.1324 | 1.2616 |
| SRO-FA historical partition | 800 | 0.01 | 10 | 0.0813 | 0.0815 | 0.0806 | 0.0820 |
| SRO-FA historical solver | 800 | 0.01 | 10 | 0.3144 | 0.3158 | 0.2632 | 0.3436 |
| SRO-FA historical OOS | 800 | 0.01 | 10 | 0.1764 | 0.1793 | 0.1629 | 0.1895 |
| SRO-FA historical total | 800 | 0.01 | 10 | 1.2219 | 1.2234 | 1.1609 | 1.2653 |
| SRO-FA historical partition | 800 | 0.02 | 10 | 0.0828 | 0.0813 | 0.0801 | 0.0939 |
| SRO-FA historical solver | 800 | 0.02 | 10 | 0.3348 | 0.3315 | 0.2825 | 0.3866 |
| SRO-FA historical OOS | 800 | 0.02 | 10 | 0.1741 | 0.1704 | 0.1661 | 0.1872 |
| SRO-FA historical total | 800 | 0.02 | 10 | 1.2699 | 1.2635 | 1.1940 | 1.4129 |
| SRO-FA historical partition | 800 | 0.05 | 10 | 0.0815 | 0.0816 | 0.0808 | 0.0822 |
| SRO-FA historical solver | 800 | 0.05 | 10 | 0.4131 | 0.4115 | 0.3688 | 0.4749 |
| SRO-FA historical OOS | 800 | 0.05 | 10 | 0.1756 | 0.1723 | 0.1685 | 0.1891 |
| SRO-FA historical total | 800 | 0.05 | 10 | 1.4224 | 1.4224 | 1.3413 | 1.5122 |
| SRO-FA historical partition | 800 | 0.1 | 10 | 0.0816 | 0.0816 | 0.0811 | 0.0825 |
| SRO-FA historical solver | 800 | 0.1 | 10 | 0.5953 | 0.5987 | 0.4592 | 0.7172 |
| SRO-FA historical OOS | 800 | 0.1 | 10 | 0.1789 | 0.1818 | 0.1716 | 0.1851 |
| SRO-FA historical total | 800 | 0.1 | 10 | 1.7453 | 1.7547 | 1.6064 | 1.8703 |
| SRO-FA historical partition | 800 | 0.15 | 10 | 0.0819 | 0.0821 | 0.0808 | 0.0824 |
| SRO-FA historical solver | 800 | 0.15 | 10 | 0.7878 | 0.7560 | 0.6795 | 1.0080 |
| SRO-FA historical OOS | 800 | 0.15 | 10 | 0.1781 | 0.1799 | 0.1696 | 0.1858 |
| SRO-FA historical total | 800 | 0.15 | 10 | 2.0918 | 2.0848 | 1.9671 | 2.3455 |
| SRO-FA historical partition | 800 | 0.2 | 10 | 0.0822 | 0.0816 | 0.0809 | 0.0894 |
| SRO-FA historical solver | 800 | 0.2 | 10 | 1.0307 | 0.9998 | 0.9077 | 1.2110 |
| SRO-FA historical OOS | 800 | 0.2 | 10 | 0.1751 | 0.1735 | 0.1663 | 0.1827 |
| SRO-FA historical total | 800 | 0.2 | 10 | 2.5035 | 2.4500 | 2.3434 | 2.7248 |
| SRO-FA historical partition | 800 | 0.3 | 10 | 0.0822 | 0.0822 | 0.0816 | 0.0828 |
| SRO-FA historical solver | 800 | 0.3 | 10 | 2.2836 | 2.2594 | 2.0282 | 2.5268 |
| SRO-FA historical OOS | 800 | 0.3 | 10 | 0.1785 | 0.1803 | 0.1697 | 0.1867 |
| SRO-FA historical total | 800 | 0.3 | 10 | 4.1049 | 4.0951 | 3.8614 | 4.3318 |
| SRO-FA historical partition | 800 | 0.4 | 10 | 0.0830 | 0.0823 | 0.0812 | 0.0912 |
| SRO-FA historical solver | 800 | 0.4 | 10 | 6.4534 | 6.4156 | 4.6311 | 7.5659 |
| SRO-FA historical OOS | 800 | 0.4 | 10 | 0.1795 | 0.1811 | 0.1722 | 0.1848 |
| SRO-FA historical total | 800 | 0.4 | 10 | 8.7401 | 8.6679 | 6.8800 | 9.8677 |
| SRO-FA historical partition | 800 | 0.45 | 10 | 0.0825 | 0.0825 | 0.0818 | 0.0832 |
| SRO-FA historical solver | 800 | 0.45 | 10 | 11.6861 | 12.6231 | 5.9085 | 16.3345 |
| SRO-FA historical OOS | 800 | 0.45 | 10 | 0.1787 | 0.1796 | 0.1702 | 0.1838 |
| SRO-FA historical total | 800 | 0.45 | 10 | 14.2288 | 15.1067 | 8.6837 | 18.8431 |
| SRO-FA historical partition | 800 | 0.5 | 10 | 0.0826 | 0.0827 | 0.0819 | 0.0831 |
| SRO-FA historical solver | 800 | 0.5 | 10 | 10.4512 | 10.2712 | 8.5418 | 13.2322 |
| SRO-FA historical OOS | 800 | 0.5 | 10 | 0.1801 | 0.1806 | 0.1704 | 0.1889 |
| SRO-FA historical total | 800 | 0.5 | 10 | 13.2643 | 13.1181 | 11.3196 | 16.0442 |
| SRO-FA historical partition | 800 | 0.6 | 10 | 0.0828 | 0.0828 | 0.0817 | 0.0849 |
| SRO-FA historical solver | 800 | 0.6 | 10 | 30.1430 | 29.8197 | 23.4882 | 36.3548 |
| SRO-FA historical OOS | 800 | 0.6 | 10 | 0.1802 | 0.1811 | 0.1734 | 0.1842 |
| SRO-FA historical total | 800 | 0.6 | 10 | 33.6305 | 33.2909 | 27.0292 | 39.9061 |
| SRO-FA historical partition | 800 | 0.75 | 10 | 0.0862 | 0.0832 | 0.0825 | 0.1035 |
| SRO-FA historical solver | 800 | 0.75 | 10 | 125.6697 | 120.6311 | 111.8749 | 156.2434 |
| SRO-FA historical OOS | 800 | 0.75 | 10 | 0.1830 | 0.1835 | 0.1712 | 0.1949 |
| SRO-FA historical total | 800 | 0.75 | 10 | 130.4370 | 125.4948 | 116.7054 | 160.9343 |
| SRO-FA historical partition | 800 | 0.8 | 10 | 0.0961 | 0.0850 | 0.0830 | 0.1407 |
| SRO-FA historical solver | 800 | 0.8 | 10 | 226.0894 | 228.9273 | 169.4870 | 258.4278 |
| SRO-FA historical OOS | 800 | 0.8 | 10 | 0.2144 | 0.1980 | 0.1855 | 0.2916 |
| SRO-FA historical total | 800 | 0.8 | 10 | 232.3574 | 235.5817 | 174.6041 | 266.6355 |
| SRO-FA historical partition | 800 | 1.0 | 10 | 0.0845 | 0.0841 | 0.0833 | 0.0876 |
| SRO-FA historical solver | 800 | 1.0 | 10 | 602.3212 | 564.8842 | 460.8009 | 913.3432 |
| SRO-FA historical OOS | 800 | 1.0 | 10 | 0.1802 | 0.1819 | 0.1709 | 0.1857 |
| SRO-FA historical total | 800 | 1.0 | 10 | 610.2055 | 572.4292 | 468.5594 | 922.2318 |
| SRO-FA historical partition | 800 | 1.5 | 10 | 0.0865 | 0.0861 | 0.0844 | 0.0937 |
| SRO-FA historical solver | 800 | 1.5 | 10 | 5385.5819 | 4776.8780 | 3895.8990 | 7894.0551 |
| SRO-FA historical OOS | 800 | 1.5 | 10 | 0.1752 | 0.1741 | 0.1618 | 0.1852 |
| SRO-FA historical total | 800 | 1.5 | 10 | 5404.4400 | 4795.6950 | 3913.5167 | 7913.6021 |
| SAA-kNN historical selection/CV (not established here) | 50/200/800 | all | 0 | unavailable | unavailable | unavailable | unavailable |

SRO-FA totals/partition/evaluation use historical time.time; time_lp is saved solver time,
not full optimization wrapper elapsed. No historical strict-status certificate is invented.
Original cache runtime fields, SRO seed records, nominal epochs, and AP/MP iteration times are retained
in the manifest and its frozen inputs. No new training, benchmark construction, or CV was performed.

## Outputs

Published figures: ['/data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/production_v1/published/comparison_R10.png', '/data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/production_v1/published/comparison_R10.pdf']; plot data: /data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/production_v1/published/plot_data.json.
runtime_aggregates.json contains current and persisted historical timings separately.
summary.json contains compact per-policy result paths and phase clocks; full policies are not rewritten per result.

SCS membership: 351400 positive residual observations; 188 above 1e-9; maximum 7.90904410941e-06.

Three historically nonconverged SCS N50 policies are included unchanged: (rho, seed) = (.15, 4), (.6, 5), (.8, 1).
Eleven green returned checkpoints have their own gap >=1%; the stopping candidate passed, and the saved minimum-AP selection is retained without reselection.
Test arrays have shape 10000 x 10. Generator output follows the original code protocol, but is not byte-certified against historical SRO-FA tests. No cached R3 test hashes are used.
Missing historical kNN selection, benchmark construction and evaluation timings are unavailable; reruns would be required. Historical SRO-FA partition, time_lp solver, evaluation and total fields remain separate; time_lp is not optimization wrapper time.

## Fresh common gray reference: correction to inherited report

Any inherited statement that no benchmark construction occurred does not apply to this R10 gray reference. It is ONE fresh joint LP fitted to the common 10,000-row test (seed 9999), shown identically in all three panels, not eight independent results, policy training, deployable OOS performance, or a population lower bound. Historical gray entries are not plotted, filled, or overwritten. No oracle solution enters neural replay, policy selection, or training.

| Common oracle scope | Status | Wall seconds | Human time |
|---|---|---:|---|
| construction | completed | 6.528310 | 6.528310 s / 0.108805 min / 0.001813 h |
| optimization | completed | 41.323418 | 41.323418 s / 0.688724 min / 0.011479 h |
| cost_replay_and_solution_serialization | completed | 0.164983 | 0.164983 s / 0.002750 min / 0.000046 h |
| replay separately (unmeasured) | unavailable | unavailable | unavailable; rerun with instrumentation required |
| serialization separately (unmeasured) | unavailable | unavailable | unavailable; rerun with instrumentation required |
| oracle.run total (excludes final result JSON) | completed | 116.078447 | 116.078447 s / 1.934641 min / 0.032244 h |
| controller total (includes result JSON, audit/binding) | completed | 116.444485 | 116.444485 s / 1.940741 min / 0.032346 h |

Oracle: one worker, one solver thread; environment and solver iterations/status are recorded in its raw result when reached. Oracle phases are separate from fresh neural evaluation and persisted nominal/CCG/CV/SRO-FA tables above. Failed or unattempted oracle timings are not completed-run aggregates.

Final status: completed; whole pipeline 952.403988 s (952.403988 s / 15.873400 min / 0.264557 h); final JSON serialization excluded.
