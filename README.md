# PROMISE

Companion code for "PROMISE: Preconditioned Stochastic Optimization
Methods by Incorporating Scalable Curvature Estimates".
We provide detailed instructions for reproducing each plot in the paper.

> :memo: **Note**<br>
We plan to create a version of PROMISE that is easy for practitioners to use in the near future!

## Preliminary steps

1. Download the required datasets to a new folder `data` by running `python download_data.py`. 
You may have to use the command `data_fixes.sh ./data` to fix some issues in the higgs, susy, and webspam datasets.
To use the yelp dataset in the showcase experiments , please visit [https://www.yelp.com/dataset](https://www.yelp.com/dataset) and preprocess the dataset using `yelp_preprocessing.py`.

2. Remove the folders `performance_results`, `suboptimality_results`, `showcase_results`, `streaming_results`, `sensitivity_results`, `spectra_results`, and `regularity results`.

After completing the preliminary steps, we can run shell scripts (`.sh`) in the `config` folder and notebooks (`.ipynb`) in the `plotting` folder to generate the plots.

## Performance experiments (Section 6.1)
First, please run `sklearn_opt_least_squares.sh`, `sklearn_opt_logistic.sh`, `sklearn_opt_least_squares_mu_1e-1.sh`, and `sklearn_opt_logistic_mu_1e-1.sh`.

Then run `performance_exp_least_squares.sh`, `performance_exp_logistic.sh`, `performance_exp_least_squares_mu_1e-1.sh`, and `performance_exp_logistic_mu_1e-1.sh`.

Once these scripts are finished running, please run `performance_results_plots.ipynb`.

## Suboptimality experiments (Section 6.2)
Please run `suboptimality_exp_least_squares.sh` and `suboptimality_exp_logistic.sh`. 

Once these scripts are finished running, please run `suboptimality_results_plots.ipynb`.

## Showcase experiments (Section 6.3)
Please run `showcase_exp_url.sh`, `showcase_exp_yelp.sh`, and `showcase_exp_acsincome.sh`.

Once these scripts are finished running, please run `showcase_results_plots.ipynb`. This notebook will also generate Figure 1 in the Introduction (Section 1).

## Streaming experiments (Section 6.4)
Please run `streaming_exp_logistic_higgs.sh` and `streaming_exp_logistic_susy.sh`. 

Once these scripts are finished running, please run `streaming_results_plots.ipynb`.

## Sensitivity study (Section 6.5)
Please run `sensitivity_exp_least_squares.sh`, `sensitivity_exp_logistic.sh`, and `compute_spectra.sh`.

Once these scripts are finished running, please run `sensitivity_results_plots.ipynb` and `spectrum_results_plots.ipynb`.

## Regularity study (Section 6.6)
Please run `regularity_exp_logistic.sh`.

Once this script is finished running, please run `regularity_results_plots.ipynb`.

> :warning: **Warning**<br>
Running all the experiments can take a lot of time. 
If you would like to generate plots based on existing results in `performance_results`, `suboptimality_results`, `showcase_results`, `streaming_results`, `sensitivity_results`, `spectra_results`, and `regularity results`, just run the corresponding notebooks mentioned above.
