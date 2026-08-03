# Tabnetics Diakrino Campaign Results

## Scope

Tabnetics Diakrino is a trained feature-ranking sidecar that plugs into the
Tabnetics feature-selection pipeline. It proposes a bounded number of feature
additions on top of a protected classical core, and everything downstream — the
classifier, the split policy, and the evaluation protocol — stays paired across
arms. What is measured here is that integration: how much the ranker adds to the
existing pipeline, not how it performs as a stand-alone classifier.

## Final campaign

The closeout run covers 64 HDLSS datasets, three paired seeds (`29`, `47`, and
`83`), and eight arms, for 1,536 reconciled result rows. The headline profile
(P4) combines multi-view Diakrino proposals, native-null admission, and
support-only JMI arbitration; it is compared against its protected-core baseline
(B0).

P4 beats B0 by a mean balanced-accuracy margin of `+0.0088`, winning on 33
datasets, tying on 14, and losing on 17. At a Holm-adjusted p-value of `0.0828`
it clears both the strict `p < 0.10` headline gate and the tier-harm gate.

The equal-addition diagnostics are the more interesting part of the picture, and
they are deliberately descriptive rather than promotion-blocking. Against the
best classical alternative, P4 gains `+0.0076` balanced accuracy (Holm `0.3562`);
against seeded random extra features, `+0.0055` (Holm `0.8521`); and against its
own ranks after random permutation, `-0.0032` (Holm `0.9383`). In other words,
the protected augmentation profile as a whole holds up, but these results do not
establish that the ranker is what produces the improvement. That is why Diakrino
ships opt-in rather than as a default.

## Reproducibility

The result matrix was reconciled against source, input, split, feature-order,
checkpoint, bundle, budget-authority, and host-telemetry identities, so every
reported row is traceable to the run that produced it. Any future claim for
Diakrino requires fresh emissions bound to their source under the same contract.

This page records campaign evidence. A peer-reviewed article describing the work
is in preparation.
