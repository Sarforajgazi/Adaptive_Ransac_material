import sys
import os
import numpy as np
from ransac_env import RansacEnv

def test_environment():
    print("Initializing RansacEnv...")
    try:
        env = RansacEnv()
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return False

    print("Checking Observation Space:", env.observation_space)
    print("Checking Action Space:", env.action_space)

    print("\n--- Resetting Environment ---")
    obs, info = env.reset()
    print(f"Loaded File: {info['file']}")
    print(f"Observation (Features): {obs}")

    print("\n--- Taking Random Actions ---")
    for step_num in range(1, 4):
        action = env.action_space.sample()
        print(f"\nStep {step_num}: Action {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print(f"Terminated: {terminated}")
        
        if terminated or truncated:
            print(f"Episode finished. Resetting...")
            obs, info = env.reset()

    print("\nEnvironment test passed!")
    return True

if __name__ == "__main__":
    test_environment()
