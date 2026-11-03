# Pre-registration (written before any test result was computed)

Unit of analysis = **one formation** (never image pairs, never circle pairs). Primary unit counts once.
Data: overhead aerial photos + diagrams (`kind` in {aerial w/ OH tag, graphic}); ground shots excluded from metric tests.
Split: `ingest.py` assigns each formation to `tune` (sha256(url) % 5 == 0, ~20%) or `locked` (~80%). Tools are developed
and tuned on `tune` only. `locked` is readable only after `analysis/FROZEN` exists (commit hash of tools recorded in it).
Dates, places, archive labels (genuine/manmade), crop type are **not read** until step 5.

## Fixed constants K (relative ratio r >= 1 of two radii in one formation)
phi=1.6180340, sqrt2, sqrt3, sqrt5, 2/sqrt3, 3/2, 4/3, 5/4, 2, 3. (|K| = 10). Match iff |r-k|/k <= delta.
**delta = 0.02**, valid only if synthetic-recovery measurement error sigma(r) <= 0.01; otherwise T1 is reported INCONCLUSIVE.
Formation hit = any radius ratio within the formation matches k. Statistic = share of formations with a hit, per k.

## Baselines (both must be beaten)
- B1 random sizes: same circle count, radii log-uniform within the formation's observed [r_min, r_max].
- B2 tape-measure: radii = round(u * R_max / s) * s, u~U(0,1), s in {R_max/10, R_max/20} (human round-number layout).
10,000 simulated libraries each; p = fraction of simulated shares >= observed.

## Tests
- T1 ratios (above). T2 symmetry order n (angular autocorrelation of skeleton, n in 2..24): share with n>=3 and distribution of n.
- T3 encoding capacity: number of distinct element classes (k-means on normalized ring descriptors, BIC), bits/element,
  LZMA-compressibility of the canonical descriptor string vs 1000 shuffles.
- T4 3D fits: orthographic projection of the 5 Platonic + 13 Archimedean solids; fit residual minimized over rotation (+scale).
  Report **number of rival solids within tolerance** per formation; a fit counts only if exactly one solid fits and beats the
  random-point-set baseline (same N points, random positions) at residual <= 0.03 * R_max.
- T5 perspective depth: fit 3D point sets by perspective vs flat camera; model chosen by BIC; flat wins by default.
- T6 style clustering: GMM/k-means on descriptors; structure claimed only if bootstrap ARI >= 0.6 AND silhouette >= 0.25.

## Statistics and decision rule
Effect size = rate ratio (observed share / baseline share) with 95% formation-level bootstrap CI (10,000 resamples).
Family = all (test x constant x baseline) cells; Holm-Bonferroni FWER alpha = 0.05.
A result is POSITIVE only if Holm-adjusted p < 0.05 AND CI lower bound > 1.25 against BOTH baselines.
No result is interpreted for meaning; a positive is "a measurable regularity", not a message.

## Step-3 gate (tools must pass before `FROZEN`)
Synthetic designs: planted phi ratios, binary strip, true perspective projection of a molecule, plus pure-null sets.
Required: recall >= 0.8 at a planted rate ratio of 2.0; false-positive rate <= 0.05 on nulls; radius error sigma <= 0.01.
If a test fails the gate its real-data result is reported as INCONCLUSIVE, not negative.

## Step 5 (unblinding, once)
Join dates, places, labels. Pre-listed associations: metric hit-rate / symmetry n / cluster id vs
{genuine vs admitted-manmade, region, year, crop}; chi-square or Spearman; Holm over the 4 x (#metrics) family.
