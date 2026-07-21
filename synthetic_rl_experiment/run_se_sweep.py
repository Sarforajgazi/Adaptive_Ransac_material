import numpy as np
from synthetic_env import SyntheticRansacEnv

def run_se_sweep():
    env = SyntheticRansacEnv(max_steps=5)
    
    # We will test the two configurations the user asked about
    configs = [
        (0.01, 0, 0.05), # Low Noise, Eps 0.05
        (0.20, 3, 0.20)  # High Noise, Eps 0.20
    ]
    norm_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    min_supp_idx = 2 # 75
    
    print('Starting n=250 SE sweep...')
    for noise, eps_idx, eps_val in configs:
        print(f'\n=== Noise: {noise:.2f} | Eps: {eps_val:.2f} ===')
        
        results = {nth: [] for nth in norm_thresholds}
        
        for seed in range(250):
            # To do MATCHED seeds, we must evaluate ALL thresholds for the SAME seed
            # before moving to the next seed! (Even though RANSAC is unseedable, 
            # this perfectly pairs the point clouds).
            for nth in norm_thresholds:
                env.generator.rng = np.random.default_rng(seed)
                env.noise_sigma = noise
                env.inlier_ratio = 0.50
                env.reset()
                
                nth_idx = norm_thresholds.index(nth)
                action = np.array([eps_idx, min_supp_idx, nth_idx, 1])
                obs, reward, done, _, info = env.step(action)
                results[nth].append(info['score'])
                
        for nth in norm_thresholds:
            scores = results[nth]
            avg = np.mean(scores)
            se = np.std(scores) / np.sqrt(len(scores))
            print(f'Norm {nth:.2f} | Score: {avg:.4f} +/- {se:.4f}')

run_se_sweep()
