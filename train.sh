python -m src.train \
  --data data/vc_real_dataset.npz \
  --out runs/vc_real \
  --dim 2 \
  --epochs 200 \
  --batch 4 \
  --lambda_l1 100 \
  --lambda_grad 10 \
  --lambda_bc 20 \
  --lambda_residual 1
