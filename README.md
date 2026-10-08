# Reaction network inference with spike-and-slab priors (WIP)

Please refer to the README under `insect`.

# Bayesian model selection of differential equation models under parameter non-identifiability

This repository hosts the code for reproducing the results of the manuscript "Reliable model selection in the presence of parameter non-identifiability" ([arXiv preprint](https://arxiv.org/abs/2605.19807)). Bayesian model selection is performed by computing model posterior probabilities exhaustively, which requires the evidence of each model to be approximated deterministically or estimated with Monte Carlo methods. The following methods for computing model evidence are implemented:

 - Bayesian information criterion (BIC),
 - Laplace importance sampling (Laplace IS),
 - standard adaptive multiple importance sampling (standard AMIS),
 - robust adaptive multiple importance sampling (robust AMIS), and
 - bridge sampling.

<!-- **N.B.** This code repository is not compatible with versions of [PEtab.jl](https://github.com/sebapersson/PEtab.jl) beyond v3.11.3, due to breaking changes in [how PEtab.jl v4 handles parameter transformations](https://github.com/sebapersson/PEtab.jl/blob/main/HISTORY.md). This compatibility requirement is reflected in the `Project.toml` of this repository. -->

## Coral re-growth example

The example in Section 5.1 is adapted from [Simpson et al. (2022)](https://doi.org/10.1016/j.jtbi.2021.110998). Model selection is performed over three sigmoidal growth models based on 11 time series observations of coral re-growth. The results are obtained by running the following scripts.

1. Running `coral/download_data.jl` downloads the dataset from the Github repository [SigmoidGrowth](https://github.com/ProfMJSimpson/SigmoidGrowth) and saves it as `coral/data.xlsx`.

2. Running `coral/fit_MAP.jl` finds the maximum *a posteriori* estimate and its corresponding Hessian for all three models and saves the results in `coral/output/MAPs.jld2`.

3. Running `coral/laplace_IS.jl` performs Laplace IS for all three models. The script takes in the repeat index `[n]` (1 to 100) as a command line argument, and saves the result for model `[m]` in `coral/output/seed[n]/laplace_IS_[m].jld2`. Standard AMIS and robust AMIS are run similarly using the scripts  `coral/orig_AMIS.jl` and `coral/robust_AMIS.jl` instead.

4. Running `coral/MCMC_chains.jl` runs MCMC to sample from the parameter posterior distributions of all three models. The script takes in the repeat index `[n]` (1 to 100) as a command line argument, and saves the result for model `[m]` in `coral/output/seed[n]/chains_[m].jld2`. 

5. Running `coral/bridge_sampling.jl` post-processes the MCMC samples for bridge sampling. The script takes in the repeat index `[n]` (1 to 100) as a command line argument, and saves the result for model `[m]` in `coral/output/seed[n]/BS_[m].jld2`. 

6. The scripts `coral/plot_logistic_richards.jl`, `coral/plot_errors_K.jl`, `coral/plot_data.jl`, `coral/plot_indiv_posteriors.jl` reproduce Figure 1, Figure 2, Figure S1, Figure S2, respectively.

## Insect life-stage example

The example in Section 5.2 features a hypothetical insect with egg, larva, and adult life stages. Each possible model consists of three core mechanisms (life stage transition and reproduction) and a subset out of six candidate death mechanisms. Model selection is performed over $2^6=64$ possible models using datasets of 41 time series observations each. The results are obtained by running the following scripts.

1. Running `insect/generate_datasets.jl` generates 44 synthetic datasets of the populations of each life stage. Ground truth parameters are tuned such that they have an order of magnitude close to $10^0$, the trajectories of all datasets end at a similar state, and the final-time derivatives are not too large. Each dataset corresponds to a different model; 20 models are excluded as they are unable to sustain a stable positive equilibrium. The ground truth parameters are saved in `insect/params.jld2`, while the datasets are saved in `insect/data.jld2`.

2. Running `insect/fit_MAP.jl` finds the maximum *a posteriori* estimate and its corresponding Hessian for all 64 models. The script takes in the dataset index `[d]` (1 to 44) as a command line argument, and saves the result (all models) for dataset `[d]` in `insect/output/data[d]/MAP.jld2` and `insect/output/data[d]/MAP_hess.jld2`.

3. Running `insect/laplace_IS.jl` performs Laplace IS for all 64 models. The script takes in the dataset index `[d]` (1 to 44) as a command line argument, and saves the result (all models) for dataset `[d]` and model `[m]` in `insect/output/data[d]/laplace_IS_model[m].jld2`. Standard AMIS and robust AMIS are run similarly using the scripts  `insect/orig_AMIS.jl` and `insect/robust_AMIS.jl` instead.

4. Running `insect/MCMC_chains.jl` runs MCMC to sample from the parameter posterior distributions of all 64 models. The script takes in the dataset index `[d]` (1 to 44) as a command line argument, and saves the result (all models) for dataset `[d]` and model `[m]` in `insect/output/data[d]/chains_model[m].jld2`.

5. Running `insect/bridge_sampling.jl` post-processes the MCMC samples for bridge sampling. The script takes in the dataset index `[d]` (1 to 44) as a command line argument, and saves the result (all models) for dataset `[d]` and model `[m]` in `insect/output/data[d]/BS_model[m].jld2`.

6. The scripts `insect/plot_comparison.jl`, `insect/plot_model_posterior.jl`, `insect/plot_data.jl`, `insect/plot_param_posteriors.jl` reproduce Figure 3, Figure 4, Figure S3, Figure S4, respectively. The variables named `LOAD_XXX` in `insect/plot_comparison.jl` should be set to `false` when running the script for the first time.