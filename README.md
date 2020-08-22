# Analysis of Multivariate Scoring Functions for Automatic Unbiased earning to Rank
In this repository, we have uploaded the code corresponding to the CIKM2020 Paper "Analysis of Multivariate Scoring Functions for Automatic Unbiased Learning to Rank", which is also on arxiv (https://arxiv.org/abs/2008.09061).

You may play with toy data we prepared by run the following command line.
1. ./All_experiment.sh 

For Yahoo and Istella-s datasets, you need to prepare them in TREC form first.The Paper contains 2 figures and 2 tables. You can reproduce all of them.
steps:
1. git clone https://github.com/ULTR-Community/ULTRA.git
2. Prepare Yahoo and Istella-s acoording to pipelines in  ULTRA/example/Istella-S and ULTRA/example/Yahoo
3. Copy the tmp_data folder from step 2 to datasets. The structure should be like datasets/toy_data.
4. Uncomment corresponding part in All_experiment.sh
5. Open demo.ipynb for plotting.
