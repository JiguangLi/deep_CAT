# Deep Computerized Adaptive Testing (CAT)

This repo contains the official Python implementation of our proposed Deep CAT system, as illustrated in our paper *Deep Computerized Adaptive Testing*: (https://arxiv.org/pdf/2502.19275). The underlying latent variable model is assumed to be the two-parameter Bayesian Multidimensional Item Response Theory (MIRT) Model. 

In additional to our Deep Q-learning approaches, we also implement common Bayesian item selection rules as described in Section 3.2 of the paper (such as Maximizing Mutual Information). Our approaches can direct sample from the latent factor posterior distributions, thereby eliminating the need for computationally expensive MCMC sampling, which cannot be easily pararalized and requires additional tuning steps.

# What is CAT?
It is an adaptive testing system primarily used in behavior health (such as detecting cognitive impaitment) or in educational assessment (think of GRE).
Unlike traditional linear tests, which present a fixed set of items to all test-takers, CAT dynamically
selects questions from a large item bank based on an examinee’s prior responses. This
adaptivity enhances both the efficiency and precision of ability estimation by continuously
presenting items that are neither too easy nor too difficult. By focusing on items
that provide the most information about a test-taker’s latent traits, CAT reduces the
number of questions needed to reach an accurate assessment, often enabling earlier test
termination without sacrificing measurement accuracy. This efficiency is especially valuable
in high-stakes diagnostic settings, such as clinical psychology, where CAT can serve as an
alternative to in-person evaluations, helping to expand access to assessments in resource limited environments.

# Why Using Our CAT System?

- We propose a double deep Q-learning algorithm to learn the optimal item selection policy from a given item bank. Experiments have shown our Q-learning approach leads to much faster posterior variance reduction, and enables earlier termination. Thinking from a reinforcement learning perspective is essential as existing item selection rules have three main limitations: 
  - (1) they rely on one-step lookahead greedy optimization of an information-theoretic criterion. While easy to implement, such myopic strategies fail to account for subsequent decisions, leading to suboptimal adaptive testing policies.
  - (2) they do not direct minimize the number of items required to terminate a test.
  - (3) they are heuristically designed to balance information across all latent factors, unable to prioritize main factors of interests.

- Even If you are uncomfortable of RL or using neural network for adaptive testing, our approach still significantly accelerate the existing common Bayesian item selection rules discussed in Section 3.2 of the paper, or any Bayesian approach that involves sampling from the latent factor posterior distributions. 

# Repo Directories

To import the source code in ```src/```, do

```bash
python setup.py install
pip install -e .
```
Then in your python script editor:
```python
import bayesian_cat as bcat
``` 

The following directories are especially helpful:

- ```environment.yaml```: to recreate our conda environment
- ```src/bayesian_cat/CAT/bayesian_CAT.py```: contains implementation of existing multivariate CAT item selection rules. Modify the ```selection_criterion``` argument to change the rule. For instance, the item selection rules considered in the paper include ```kl_eap```,```kl_pos```, ```mi_sir```, ```predictive_variance_e```.
- ```src/bayesian_cat/QCAT```: contains the deep Q-learning CAT online deployment ```deep_Q_CAT.py```, neural network architectures ```deep_q_network.py```, episode object during Q-learning ```episode_learner```, and other helper files during Q-learning such as replay buffer.
- ```src/bayesian_cat/FullyBayesianCAT```: contains the fully-Bayesian version of the item selection rules that also incorporate item parameter uncertainties. See appendix of the paper.

# Experiment Replications


1. **Section 6 (simulation)**: Navigate to the ```project/standarized_simulation``` folder:

   - Run script ```s01_generate_sim_params.py``` to generate simulation parameters. Here we generated 150 item, 5-factor item parameters. The goal is to accurately assess the first 3 factors while accounting for the prescence of factor 4 and 5.
   - Run script ```s02_fit_non_rl_models.py``` to run existing benchmarks (Optional). This includes "KL-EAP", "KL-POS", "MI", "Max Var" in the paper. These methods involve integrating over all $5$ factors or interests. Note that after paper revision, we no longer use these benchmarks in the paper, but used the model outputs from ```s05_fit_non_rl_subset.py``` instead.
   - Run script ```s03_online_deep_q_learning.py``` to run double deep Q-learning algorithm. Expect 2-3 days on a single GPU. The script would output the final neural networks for item selection.
   - Run Script ```s03_evaluate_online_q_network.py``` to take the neural network as input, and evaluate its performance from simulated online subjects.
   - Run script ```s05_fit_non_rl_subset.py``` to run existing benchmarks for "KL-EAP", "KL-POS", "MI", "Max Var". The key difference from ```s02_fit_non_rl_models.py``` is that now we only integrated over the first 3 factors of interests while still accounting for the prescence of factors 4 and 5. Given that we wish to accurately assess the first 3 factors,  this is a stronger benchmark than ```s02_fit_non_rl_models.py``` as pointed out by our reviewers. Hence we use model outputs from this script as benchmarks in our paper after revision.
   - We provide optional script ```s05_retrain_online_deep_q_learning.py``` to continue finetuning the neural networks from ```s03_online_deep_q_learning.py```. This script is not required.

2. **Section 7 (Cognitive Assessment)**: Navigate to the ```project/cat_cog_experiment``` folder:

   - Run script ```s01_fit_bifactor_model.py``` to obtain item bank item parameters. Alternatively, you can find the items parameters already stored in ```models/cat_cog_models/bifactor_alphas.feather```. Note for this experiemnt, we have a six factor model, but we are interested in the primary cognitive factor, while still taking into account additional 5 factors during experiment.
   - Run sript ```s02_fit_non_rl_models.py``` (optional) to run existing benchmarks for "KL-EAP", "KL-POS", "MI", "Max Var". Note for the same reasons as in simulation, we did not use these benchmarks after the revision since they are standard baseline integrating over all 6 factors.
   - Run script ```s03_online_deep_q_learning.py``` to run double deep Q-learning algorithm. Expect 2 days on a single GPU.
   - Run script ```s05_evaluate_online_q_network.py``` to evaluate Q-learning.
   - Run script ```s07_fit_non_rl_subset.py``` for baseline methods: "KL-EAP", "KL-POS", "MI", "Max Var". As in Section 6, we modified the standard baseline by only integrating over the primary factors of interest while still accounting for the prescence of additional 5 factors. We use model outputs from this script after the revision.


   # Table, Figures, and More Visualization.

   - Section 6: To create every figures, statistics, and tables in section 6 of the paper, find the jupyter markdown files in ```markdowns/simulation_markdown_revision/paper_version_5factor_first3.ipynb```. Some evaluation codes may take upto 1 hour to run, but we have commented them out and saved the relevant outputs in the same directory for easier replications. But do feel free to uncomment them out and rerun from scratch.

   - Section 7: To create every figures, statistics, and tables in section 7 of the paper, find the jupyter markdown files in ```markdowns/cat_cog_markdown_v2/paper_version_new.ipynb```. Same as above, some evaluation codes may take upto 1 hour to run, but we have commented them out and saved the relevant outputs in the same directory for easier replications. 

   - We have also uploaded relevant data, baseline models, neural network outputs in the repo. They are in the ```data``` and ```model``` folders respectively.

