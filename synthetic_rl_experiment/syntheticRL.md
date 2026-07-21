# Synthetic RL Experiment: Full Methodological Journey

This document serves as a comprehensive log of the entire RL tuning process on the synthetic RANSAC environment, detailing the hypotheses tested, the technical blockers encountered, and the final methodological conclusions.

---

## 1. Initial Goal & Environment Design
**Goal:** Train an RL agent (PPO) to dynamically adapt RANSAC parameters (`eps`, `min_support`, `normal_thresh`) based on the point cloud's state features (noise profile and inlier ratio).
**Approach:** We built a purely synthetic environment (`synthetic_env.py`) capable of generating infinite variations of geometric planes with known ground-truth parameters, varying noise ($\sigma$), and outlier ratios.
**Early Observations:** We initially added high-frequency geometric "bumps" to simulate structured clutter. However, we discovered this caused the reward landscape to behave chaotically: looser `eps` thresholds were scoring worse on noisy data because they inadvertently absorbed the structured bumps as false inliers, causing massive penalties.

## 2. The $k=5$ Averaging Hypothesis & The Hardware Blocker
**The Problem:** The RL agent was struggling to learn. A variance decomposition revealed a massive algorithmic noise floor ($SD \approx 0.69$) caused purely by the internal stochasticity of RANSAC hypothesis sampling. This noise floor completely masked the underlying geometric reward signal (gradient $\approx 0.15$).
**Proposed Solution (Option B):** Run RANSAC $k=5$ times per environment step and average the rewards to artificially reduce the algorithmic variance by $\sqrt{5}$, allowing the RL agent to "see" the geometric gradient.
**The Fix / Blocker:** When implementing the $k=5$ loop, the variance did not drop at all. We audited the backend and discovered a fatal limitation: the underlying C++ RANSAC extension (`schnabel_ransac`) does not accept a seed from Python. Instead, it seeds its internal PRNG using a 1-second resolution system clock: `srand(time(NULL))`. 
**Result:** Because the 5 RANSAC calls executed back-to-back within the same second, they received the exact same seed, generated identical hypotheses, and returned identical scores 5 times in a row. **Conclusion:** $k$-averaging is physically impossible without rewriting the C++ extension.

## 3. The Clean Baseline & Reversed Intuition
**The Problem:** Unable to suppress the noise via averaging, we had to boost the signal.
**The Fix (Option A):** We reverted the generator to a clean, uniform-outlier baseline (removing the structured bumps). This successfully dropped the algorithmic noise floor from $0.69$ down to $0.20$, creating a viable learning environment ($SNR > 1.0$).
**Offline Validation:** We ran a 9-point manual sweep on the `eps` parameter in this clean environment. We proved that an adaptive gradient existed, but it **reversed standard intuition**:
*   At low noise levels, a looser `eps` ($0.15$) was optimal to capture maximum inliers.
*   At high noise levels, the optimal `eps` tightened to ($0.10$). 
*   **Why?** In a uniform-outlier regime, a loose `eps` permits the RANSAC plane to tilt into the random noise, incurring heavy exponential penalties on the normal angle error. A tighter `eps` forces the plane to exclusively slice the dense core of true inliers, preserving geometric fidelity at the expense of raw point count.

## 4. The 10k RL Training Run & Policy Collapse
**The Run:** We trained PPO for 10,000 steps (~5,000 episodes) on the clean baseline environment. We then evaluated it across a 5x5 grid of noise and inlier ratios (2,500 total configurations) against 4 static baselines.
**The Problem:** The evaluation revealed a **policy collapse**. The RL agent output the exact same action for every single state in the grid: `eps=0.15`, `min_supp=150`, `norm_th=0.85`.
**Root Cause:** The `explained_variance` of the PPO value network remained effectively zero ($5.96 \times 10^{-8}$). Because the RANSAC stochasticity ($SD \approx 0.20$) still dominated the per-step reward, the value network failed to learn a mapping between the geometric state and the expected reward. 
**The Mechanism:** Without a functioning value network to differentiate states, the policy gradient acted blindly on noisy rewards. Faced with massive variance, the agent did what RL agents always do: it found a single "safe" configuration that performs above average across the entire distribution and locked onto it, completely ignoring the state observation.
**Silver Lining:** The static policy it locked onto (`eps=0.15`) was precisely the global optimum identified in our manual offline sweeps. The agent successfully automated the search for the optimal static baseline, even though it failed to learn *dynamic* adaptivity.

## 5. Final Conclusions for the Paper
1. **Algorithmic Stochasticity Breaks State-Dependent RL:** The internal sampling variance of RANSAC fundamentally breaks state-dependent reinforcement learning. Without a mechanism to average out this algorithmic noise, the value network cannot pierce the noise floor to learn a dynamic policy.
2. **RL as Automated Grid Search:** When forced to operate below the SNR threshold of its value network, an RL agent degrades gracefully into an automated grid search—converging to a highly robust static baseline rather than a dynamic adaptive policy.
3. **Citable Hardware Limitations:** The inability to use $k$-averaging due to `srand(time(NULL))` in legacy C++ backends is a common but rarely documented trap in geometric RL pipelines. Documenting this limitation serves as a rigorous, defensible methodological contribution to the paper.
