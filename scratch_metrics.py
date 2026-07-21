import pandas as pd
import glob
import os

print("Dataset, Mode, Frames, Mean IoU, Precision, Recall")
files = glob.glob("logs/*_rl_v4_model.csv") + glob.glob("logs/*_standard.csv")
for f in files:
    env = os.path.basename(f).split('_')[0]
    if env in ["Gascola", "House", "NordicHarbor", "WesternDesertTown"]:
        df = pd.read_csv(f)
        mode = "RL" if "rl" in f else "Standard Baseline"
        
        iou = df['iou'].mean() if 'iou' in df.columns else 'N/A'
        precision = df['precision'].mean() if 'precision' in df.columns else 'N/A'
        recall = df['recall'].mean() if 'recall' in df.columns else 'N/A'
        
        print(f"{env:18} | {mode:17} | {len(df):5} | IoU: {iou} | Prec: {precision} | Rec: {recall}")
