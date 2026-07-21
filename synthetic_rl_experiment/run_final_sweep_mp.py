import numpy as np
import multiprocessing as mp
from synthetic_env import SyntheticRansacEnv

def eval_config(args):
    seed, noise, eps_idx, nth = args
    env = SyntheticRansacEnv(max_steps=5)
    env.generator.rng = np.random.default_rng(seed)
    
    # Set physical parameters BEFORE reset
    env.noise_sigma = noise
    env.inlier_ratio = 0.50
    env.reset()
    
    nth_idx = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85].index(nth)
    action = np.array([eps_idx, 2, nth_idx, 1])
    obs, reward, done, _, info = env.step(action)
    return info['false_inlier_rate'], info['inlier_recovery_rate'], info['score']

if __name__ == '__main__':
    noises = [0.01, 0.10, 0.20]
    eps_vals = [0, 3] # 0.05, 0.20
    norm_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    print('Starting n=250 parallel sweep...')
    
    with mp.Pool() as pool:
        for noise in noises:
            for eps_idx in eps_vals:
                eps_val = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30][eps_idx]
                print(f'\n=== Noise: {noise:.2f} | Eps: {eps_val:.2f} ===')
                
                results = {nth: {'fi': [], 'score': [], 'rec': []} for nth in norm_thresholds}
                
                for nth in norm_thresholds:
                    tasks = [(seed, noise, eps_idx, nth) for seed in range(250)]
                    res = pool.map(eval_config, tasks)
                    for fi, rec, score in res:
                        results[nth]['fi'].append(fi)
                        results[nth]['rec'].append(rec)
                        results[nth]['score'].append(score)
                        
                for nth in norm_thresholds:
                    avg_fi = np.mean(results[nth]['fi'])
                    avg_score = np.mean(results[nth]['score'])
                    avg_rec = np.mean(results[nth]['rec'])
                    print(f'Norm {nth:.2f} | FI: {avg_fi:.4f} | Rec: {avg_rec:.4f} | Score: {avg_score:.4f}')
