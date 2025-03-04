# Deep Computerized Adaptive Testing (CAT)

This repo contains the official Python implementation of our proposed Deep CAT system, as illustrated in our paper *Deep Computerized Adaptive Testing*: (https://arxiv.org/pdf/2502.19275). The underlying latent variable model is assumed to be the two-parameter Bayesian Multidimensional Item Response Theory (MIRT) Model. 

In additional to our Deep Q-learning approaches, we also implement common Bayesian item selection rules as described in Section 3.2 of the paper (such as Maximizing Mutual Information). Our approaches can direct sample from the latent factor posterior distributions, thereby eliminating the need for computationally expensive MCMC sampling, which cannot be easily pararalized and requires

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

- ```src/bayesian_cat/CAT/bayesian_CAT.py```: contains implementation of existing multivariate CAT item selection rules. Modify the ```selection_criterion``` argument to change the rule. For instance, the item selection rules defined in equations (2), (3), (4), (5) of the paper corresponds to ```kl_eap```,```kl_pos```, ```mi_sir```, ```predictive_variance_e```.
- ```src/bayesian_cat/QCAT```: contains the deep Q-learning CAT online deployment ```deep_Q_CAT.py```, neural network architectures ```deep_q_network.py```, episode object during Q-learning ```episode_learner```, and other helper files during Q-learning such as replay buffer.
- ```src/bayesian_cat/FullyBayesianCAT```: contains the fully-Bayesian version of the item selection rules that also incorporate item parameter uncertainties. See Section 4.2 of the paper.

# Experiment Replications


1. **Section 6.1 (simulation)**: Navigate to the ```project/standarized_simulation``` folder:

   - Run script ```s01_generate_sim_params.py``` to generate simulation parameters.
   - Run sript ```s02_fit_non_rl_models.py``` to run existing benchmarks
   - Run Script ```project/deep_q_learning/s03_online_deep_q_learning.py``` to run double deep Q-learning algorithm. Expect 3 days on a single GPU.
   - Run script ```project/deep_q_learning/s04_evaluate_online_q_network.py``` to evaluate Q-learning
   - Run markdown ```markdowns/simulatio_markdown/paper_version_5factor_first3.ipynb``` to generate figures and tables.

2. **Section 6.2 (Cognitive Assessment)**: Navigate to the ```project/cat_cog_experiment``` folder:

   - Run script ```s01_fit_bifactor_model.py``` to obtain item bank item parameters.
   - Run sript ```s02_fit_non_rl_models.py``` to run existing benchmarks
   - Run Script ```03_online_deep_q_learning.py``` to run double deep Q-learning algorithm. Expect 2 days on a single GPU.
   - Run script ```s05_evaluate_online_q_network.py``` to evaluate Q-learning
   - Run markdown ```markdowns/cat_cog_markdown/paper_version_new.ipynb``` to generate figures and tables.

3. **Section 6.3 (Educational Assessment)**: Navigate to the ```project/dese_experiment``` folder:

   - Run script ```s02_fit_dese_data.R``` to obtain item bank item parameters.
   - Run sript ```s03_fit_non_rl_models.py``` to run existing benchmarks
   - Run Script ```04_online_deep_q_learning.py``` to run double deep Q-learning algorithm. Expect 2 days on a single GPU.
   - Run script ```s06_evaluate_online_q_network.py``` to evaluate Q-learning
   - Run markdown ```markdowns/dese_markdown/paper_dese.ipynb``` to generate figures and tables.
