# Toolkit for trace & invariant diff test

## Motive

Currently the invariant result for New, Not thorougly investigated Bugs is a bit hard to reason.
When we fail to obtain desired invariant for future input, it is necessary to localize the problem (i.e. whether it is not traced or not correctly inferred). Therefore I create toolkit to pinpoint the potential trace anomalies during differential test from original API traces running on the pre-bug, buggy, and fixed environment.

If we find the expected anomalies from the trace, then we should improve our relations to cover that. Otherwise, we should consider imporving our trace coverage.

## Functionality

Currently the tool chain could:

automatically extract the difference between two trace execution and generate a diff file
eliminate false positive and give out potential anomalies by comparing the bug<->fix diff file with the pre-bug<->fix diff file
This would help us to:

- foresee whether the current trace would be capable to infer the bug
- design what is the possible relation to infer that bug

## Usage

To use this trace analysis tool, simply go with the folllowing bash script:

### generate bug<->fix diff file
python -m traincheck.toolkit.analyze_trace -f <trace file 1> <trace file 2> -o <bug-fix-diff-file>
### generate pre-bug<->fix diff file 
python -m traincheck.toolkit.analyze_trace -f <trace file 1> <trace file 2> -o <pre-fix-diff-file>
### get rid of false positives
python -m traincheck.toolkit.detect_anomaly_from_trace_diff <bug-fix-diff-file> <pre-fix-diff-file> -o <trace_anomalies.json>
I'm currently using this tool to analyze the feasibility to infer bug LT-725 by simply using the API trace. I may further extend it to support VAR traces if needed.
