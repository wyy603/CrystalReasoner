# Human-Readable Consistency Report for Thinking-Trace Examples

This report summarizes Experiment 1.3 consistency checks after replacing
average bond-length error with three semantic structure-consistency metrics:
volume consistency, coordination environment consistency, and element-pair
presence consistency.

The aggregate statistics are computed on a small subset of 128 generated
structures with distinct `mp_id` values. The three example structures reported
below are randomly sampled from this same 128-structure subset with seed
`20260505`.

## Metric Definitions

Volume consistency compares the volume stated in the model's thinking trace
with the volume computed from the final generated CIF. A sample is marked as a
match when the relative error is at most 10%.

Coordination environment consistency parses coordination claims from the
thinking trace, such as "X is bonded to six Y atoms", "X is bonded in a
6-coordinate geometry", or polyhedral expressions such as "XO6 octahedra" and
"XO4 tetrahedra". The final CIF is then used to infer first-shell coordination:
for each element pair, the shortest periodic distance defines `d_min`, and
neighbors within `d_min + max(0.30, 0.15 * d_min)` are counted. Only the
neighbor count is compared.

Element-pair presence consistency compares element pairs mentioned in the
thinking trace with first-shell element pairs present in the final CIF. The main
score is precision over mentioned pairs, because the response is not expected to
mention every actual first-shell pair.

## Aggregate Results on 128 Distinct Structures

| Metric | Value |
|---|---:|
| Requested subset size | 128 |
| Actual subset size | 128 |
| Unique `mp_id` count | 128 |
| Structures with parsed coordination claims | 107 |
| Coordination exact-match claims | 159 / 468 |
| Coordination exact-match rate, micro | 33.97% |
| Coordination exact-match rate, macro over structures | 35.80% |
| Volume matches | 104 / 128 |
| Volume match rate | 81.25% |
| Element-pair presence precision, macro | 91.51% |

The micro coordination rate pools all parsed coordination claims before
averaging. The macro coordination rate first computes the exact-match fraction
within each structure that has at least one parsed coordination claim, then
averages over those structures.

## Random Example Summary

| Random label | MP ID | Similarity | Volume rel. error | Volume match | Coordination exact matches | Pair precision |
|---|---:|---:|---:|---|---:|---:|
| random_1 | mp-1223834 | 0.5616 | 4.97% | yes | 1 / 3 | 1.00 |
| random_2 | mp-1218433 | 0.2731 | 18.48% | no | 0 / 8 | 1.00 |
| random_3 | mp-1176451 | 0.4768 | 1.88% | yes | 0 / 6 | 1.00 |

## Example-Level Findings

### random_1: mp-1223834

The claimed volume is 121.58, while the final-CIF volume is 127.9389. The
relative error is 4.97%, so the volume check passes.

The model mentions Al-Hf and Hf-Si pairs, and both pairs are present in the
first-shell graph of the final CIF. Pair-presence precision is therefore 1.00.

Coordination consistency is mixed. The Al 10-coordinate claim matches exactly,
but the Hf 4-coordinate claim is inferred as 2-coordinate, and the Si
10-coordinate claim is inferred as 4-coordinate. The exact-match count is 1 out
of 3 parsed coordination claims.

### random_2: mp-1218433

The claimed volume is 236.30, while the final-CIF volume is 289.8543. The
relative error is 18.48%, so the volume check fails.

The model mentions Mn-O, Pr-O, and Sr-O pairs, and all three are present in the
first-shell graph of the final CIF. Pair-presence precision is 1.00.

The coordination claims are inconsistent with the final CIF. The model claims
12-coordinate Sr-O and Pr-O environments and a MnO6 octahedral environment, but
the inferred first-shell counts are much smaller for these claims. The
exact-match count is 0 out of 8 parsed coordination claims.

### random_3: mp-1176451

The claimed volume is 173.86, while the final-CIF volume is 177.1861. The
relative error is 1.88%, so the volume check passes.

The model mentions Mn-O and V-O pairs, and both are present in the first-shell
graph of the final CIF. Pair-presence precision is 1.00.

The coordination claims do not match exactly. The response claims MnO6 and VO6
octahedral environments, but the inferred first-shell counts are 4 for both
Mn-O and V-O. The exact-match count is 0 out of 6 parsed coordination claims.

## Paper-Ready Interpretation

On the 128-structure subset with distinct `mp_id` values, the generated
thinking traces show stronger consistency in scalar volume and coarse
element-pair presence than in detailed local coordination. The volume match rate
is 81.25%, and the macro element-pair presence precision is 91.51%. In contrast,
coordination exact-match accuracy is 33.97% at the claim level and 35.80% when
averaged over structures.

These results suggest that the model often preserves the major chemical
connectivity it describes, while the detailed coordination numbers in the
reasoning trace are less reliably aligned with the final generated CIF. This
supports treating consistency as a structured multi-level property: volume
agreement measures global scale, element-pair presence measures coarse
connectivity, and coordination exact match measures local environment fidelity.
No primary reported metric in this summary depends on average bond-length error.
