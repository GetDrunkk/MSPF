#!/usr/bin/env python
# ----------------------------------------------------------
# eval_infill.py ― 连续长缺口（gap_len）插值评估 + 绘图
# 用法示例：
#   python scripts/eval_infill.py \
#          --config  configs/etth_gap1000.yaml \
#          --ckpt    10 \
#          --name    etth_gap1k \
#          --plot_num 3
# ----------------------------------------------------------
import os, argparse, yaml, json, numpy as np, matplotlib.pyplot as plt, torch
from pathlib import Path
from Utils.io_utils import instantiate_from_config
from Data.build_dataloader import build_dataloader_cond
from engine.solver import Trainer

# ----------------------- CLI -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config',    required=True)
parser.add_argument('--ckpt',      type=int, required=True, help='checkpoint‑id')
parser.add_argument('--name',      required=True,          help='实验简称，用于输出目录')
parser.add_argument('--plot_num',  type=int, default=1,    help='绘图样本数')
args = parser.parse_args()

# ---------------- configuration --------------------
with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = instantiate_from_config(cfg['model']).to(device)

# -------- build TEST dataloader --------------------
# ★ 这里只需告诉 build_dataloader_cond “mode=infill”
ns = argparse.Namespace(save_dir='OUTPUT',            # dummy
                        mode='infill',
                        missing_ratio=None,
                        pred_len=None,
                        long_len=cfg['model']['params']['seq_length'])  # placeholder

dl_info   = build_dataloader_cond(cfg, ns)
test_dl   = dl_info['dataloader']
shape     = [dl_info['dataset'].window, dl_info['dataset'].var_num]

# ---------------- Trainer & checkpoint -------------
#   logger=None → 不写日志，只做推理
trainer = Trainer(cfg,
                  argparse.Namespace(name=args.name),   # dummy Args
                  model,
                  dataloader={'dataloader': test_dl},
                  logger=None)

os.environ['results_folder'] = f'./Checkpoints_{args.name}_{shape[0]}'
trainer.load(args.ckpt, verbose=True)

# ---------------- inference ------------------------
samples, reals, masks = trainer.restore(test_dl, shape=shape)
gap_region = ~masks.astype(bool)        # Bool mask – gap positions
mse  = np.mean(((samples - reals) ** 2)[gap_region])
mae  = np.mean(np.abs(samples - reals)[gap_region])

print(f'✅  Infill done:  MSE={mse:.6f}  MAE={mae:.6f}')

# -------------- save arrays & metrics -------------
out_dir = Path(f'OUTPUT/{args.name}')
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / 'samples.npy', samples)
np.save(out_dir / 'reals.npy',   reals)
np.save(out_dir / 'masks.npy',   masks)
with open(out_dir / 'metrics.json', 'w') as f:
    json.dump({'MSE': float(mse), 'MAE': float(mae)}, f, indent=2)

# -------------------- plotting ---------------------
plt.rcParams["font.size"] = 12
seq_len, D = shape
gap_mask_1d = gap_region[0,:,0]            # 所有样本同一缺口

left  = np.where(gap_mask_1d)[0][0] - 1    # 左边界前 1
right = np.where(gap_mask_1d)[0][-1]       # 右边界

for idx in range(min(args.plot_num, samples.shape[0])):
    for d in range(D):
        plt.figure(figsize=(15,3))
        # 历史 + GT 全体
        plt.plot(reals[idx,:,d],                 c='c', label='History & GT')
        # 缺口真实
        plt.plot(range(left+1, right+1),
                 reals[idx,left+1:right+1,d],    c='g', label='GT Gap')
        # 模型插值
        plt.plot(range(left+1, right+1),
                 samples[idx,left+1:right+1,d],  c='r', label='Infill')

        plt.axvline(left+1,  c='k', ls='--'); plt.axvline(right, c='k', ls='--')
        plt.title(f'seq#{idx} var#{d}  MSE={mse:.4f}')
        plt.xlabel('Time'); plt.ylabel('Value'); plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'seq{idx}_var{d}.png', dpi=150)
        plt.close()

print(f'📊  All results saved to {out_dir}')
