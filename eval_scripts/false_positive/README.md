# False Positivity Evaluation

## Goal: 
Determine the percentage of **valid** invariants inferred by TrainCheck, w.r.t at least the original input pipeline.

## Methodology:

1. Manual investigation of a small set of invariants.

2. Experimentally,
    1. apply the invariants inferred from **one input** to itself with different argument/data
    2. apply the invariants inferred from *one input* / *one class of input* to *one class of input* or different *classes of input*

## Figures to present:
1. Self-transfer invariants
    bar chart,
    each bar group corresponds to a input, and bars within a group will show different settings (e.g. different version, different optim, different size etc.)
2. Cross pipeline invariants false positivity
    x-axis: num inputs
    y-axis: fp rate


**Note** for the cross pipeline evaluation it is important to remove the compound effect of transferability.