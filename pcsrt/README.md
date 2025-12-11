# PCSRT - Progressive CSRT Tracker with Deep Features

Implementation of hybrid CSRT tracker combining traditional correlation filter tracking with deep features through learned projection.

## Architecture

**Key Components:**

1. **Feature Extraction**: VGG16/ResNet/MobileNet backbone for deep features
2. **CorrProject**: Learned projection network (512→256→64→31 channels) mapping deep features to correlation filter space
3. **DCF Solver**: Discriminative Correlation Filter with ADMM optimization for mask-constrained filter learning
4. **Mask Generation**: Binary segmentation mask from response consistency
5. **Adaptive Fusion**: Learned gating mechanism to combine `h_csrt` (traditional) and `h'_deep` (projected deep) filters

**Innovation:**
```
h_final = α·h_csrt + (1-α)·h'_deep
```
where α is adaptively learned based on tracking context.

## Installation

```bash
cd pcsrt/
pip install -r requirements.txt
```

## Training

Train on OTB100 dataset:

```bash
python train.py \
    --dataset-root ../otb100/OTB-dataset/OTB100 \
    --batch-size 8 \
    --num-epochs 50 \
    --lr 1e-4 \
    --device cuda \
    --save-dir checkpoints
```

**Training monitors:**
- TensorBoard: `tensorboard --logdir runs/`
- Loss components: peak loss, smoothness, regularization
- Adaptive α values

## Testing

Evaluate trained model on OTB benchmark:

```bash
python test.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --dataset-root ../otb100/OTB-dataset/OTB100 \
    --visualize \
    --save-results results/
```

**Outputs:**
- Per-sequence IoU, success rate, FPS
- Overall AUC score
- Success plot (saved to `results/success_plot.png`)
- Detailed results JSON

## Configuration

Edit `config.py` for hyperparameters:

```python
# Loss weights
lambda_peak = 1.0       # Peak response loss
lambda_smooth = 0.1     # Smoothness regularization  
lambda_reg = 1e-4       # Weight decay

# Feature extraction
backbone = 'vgg16'      # 'vgg16', 'resnet50', 'mobilenetv2'
feature_layer = 'conv4_3'  # VGG16 conv4_3 (28x28x512)

# Adaptive fusion
alpha_adaptive = True   # Use learned gating
alpha_min = 0.3
alpha_max = 0.9

# Update frequency
update_deep_every = 1   # Update h_deep every N frames
```

## File Structure

```
pcsrt/
├── __init__.py
├── config.py              # Hyperparameters
├── feature_extractor.py   # VGG/ResNet/MobileNet backbone
├── corr_project.py        # CorrProject + AdaptiveGating + HybridFilter
├── dcf_solver.py          # DCF with ADMM mask constraint
├── segmentation.py        # Binary mask generation
├── loss.py                # CompositeLoss (peak + smooth + reg)
├── dataset.py             # OTB dataset loader
├── tracker.py             # PCSRT main tracker
├── train.py               # Training script
├── test.py                # Evaluation script
├── utils.py               # Helper functions
└── requirements.txt       # Dependencies
```

## Testing Individual Modules

Each module has `if __name__ == '__main__'` tests:

```bash
# Test feature extractor
python feature_extractor.py

# Test CorrProject
python corr_project.py

# Test DCF solver
python dcf_solver.py

# Test loss functions
python loss.py

# Test dataset
python dataset.py
```

## Comparison with CSRT

| Metric | CSRT (baseline) | PCSRT (ours) |
|--------|----------------|--------------|
| Features | HOG + ColorNames | Deep (VGG/ResNet) + HOG |
| Mask | Binary segmentation | Binary segmentation |
| Filter | h_csrt only | α·h_csrt + (1-α)·h'_deep |
| Adaptive? | No | Yes (learned α) |

## Expected Performance

Based on OTB100 benchmark:
- **CSRT baseline**: ~0.70 AUC
- **PCSRT (target)**: ~0.72-0.75 AUC (improvement from deep features)
- **FPS**: 15-25 (depending on backbone)

## License

This implementation is for research purposes.

## Citation

If you use this code, please cite:
```
CSRT: Discriminative Correlation Filter with Channel and Spatial Reliability (CVPR 2017)
```
