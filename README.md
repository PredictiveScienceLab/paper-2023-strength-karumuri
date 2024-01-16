# paper-2023-strength-karumuri
This repository replicates the result of the paper "Hierarchical Bayesian approach to experimental data fusion: Application to
strength prediction of high entropy alloys from hardness measurements."

## **Hierarchical Bayesian approach to experimental data fusion: Application to strength prediction of high entropy alloys from hardness measurements**
[Sharmila Karumuri](https://scholar.google.com/citations?user=uY1G-S0AAAAJ&hl=en), [Zachary D. McClure](https://scholar.google.com/citations?hl=en&user=DOSWfs4AAAAJ&view_op=list_works&sortby=pubdate), [Alejandro Strachan](https://scholar.google.com/citations?user=JOeDlUkAAAAJ&hl=en), [Michael Titus](https://scholar.google.com/citations?user=QvXL-YEAAAAJ&hl=en), and [Ilias Bilionis](https://scholar.google.com/citations?user=rjXLtJMAAAAJ&hl=en).

Our paper proposes a regression methodology that can deal with input uncertainty when one wishes to correlate an inexpensive experimental measurement (e.g., hardness) to an expensive one (e.g., yield strength). Our hierarchical Bayesian approach uses two Gaussian processes. The first one maps noiseless physical descriptors to the inexpensive experimental measurement. The second Gaussian process maps noiseless physical descriptors and the inexpensive experimental measurement to the expensive experimental measurement. The two Gaussian processes form a nested model that is not analytically tractable. To overcome this issue, we propose semi-analytical approximations to both the marginal likelihood and the posterior predictive distribution. 

[Toyprob1](https://github.com/PredictiveScienceLab/paper-2023-strength-karumuri/tree/main/Toyprob1) and [HEAprob](https://github.com/PredictiveScienceLab/paper-2023-strength-karumuri/tree/main/HEAprob) contains the Python scripts that implement the examples discussed in the paper.
