# Basketball Sequence Test Results

## Dataset Info
- **Sequence**: Basketball (OTB-100)
- **Total frames**: 725
- **Test frames**: 49 (subset)
- **Resolution**: 576×432
- **Initial bbox**: [198, 214, 34, 81]

## Pure CSRT Baseline (OpenCV TrackerCSRT)

### Configuration
- Template size: 127×127 (CSRT paper default)
- Search region: 255×255
- Features: HOG (21 channels) + ColorNames (10 channels) = 31
- Response map: 255×255
- Update: Every frame

### Results
```
AUC:             0.700
Precision@20px:  1.000
Mean IoU:        0.704
Mean Distance:   5.8 pixels
```

### Implementation
- **File**: [simple_dual_tracker.cpp](src/simple_dual_tracker.cpp)
- **Tracker**: `cv::TrackerCSRT::create()` (OpenCV built-in)
- **Build**: Single file, links against `opencv_tracking`
- **Status**: ✅ **WORKING**

## Deep Feature Extraction (VGG16 → CorrProject)

### Pipeline
1. Extract 127×127 patch centered on CSRT bbox
2. VGG16 conv4_3: → 512 channels, 15×15 spatial
3. CorrProject (1×1 conv): 512 → 31 channels
4. Output: [1, 31, 15, 15] feature map

### ONNX Models Loaded
- ✅ `vgg16_conv4_3.onnx` (30.5 MB, pretrained ImageNet)
- ✅ `corr_project.onnx` (600 KB, trained 5 epochs)
- ✅ `adaptive_gating.onnx` (3 KB, trained 5 epochs)

### Current Status
- Deep features extract successfully
- **Not yet blending** - just logging for now
- Attempted bbox refinement → **WORSE** (AUC 0.658 < 0.700)

## Failed Approaches

### 1. Simple Bbox Refinement
- **Idea**: Shrink bbox by 5-10% when deep confidence < threshold
- **Result**: AUC **0.658** (worse than pure CSRT 0.700)
- **Reason**: Bbox shrinking reduces IoU without improving tracking

### 2. Adaptive Gating Network
- **Issue**: Gate network expects [1, 31] input (global pooled features)
- **Problem**: OpenCV Mat multidimensional slicing complex
- **Status**: Skipped for baseline - using pure CSRT bbox

## Next Steps

### Immediate (Response Blending)
1. Train simple DCF filter on deep features:
   - Input: 31-channel features (15×15)
   - Target: Gaussian label (15×15)
   - ADMM solver (same as CSRT)
2. Resize deep response to 255×255 (match CSRT)
3. Blend: `α·csrt_response + (1-α)·deep_response`
4. Find peak → final bbox

### Alternative (Quality-based Switching)
1. Compute CSRT response peak value (confidence score)
2. Compute deep feature quality (PCA/variance)
3. If CSRT confident → use CSRT bbox
4. If uncertain → use deep features to re-detect

### Long-term (Full OTB-100)
1. Implement response blending in C++
2. Run on all 100 sequences
3. Compare: Pure CSRT vs Dual-branch
4. Generate paper-style plots (Success vs Precision)

## Conclusions

✅ **Pure CSRT baseline validated**: AUC 0.700 is reasonable for Basketball
✅ **Deep networks working**: VGG16 → CorrProject loads and extracts features
❌ **Simple bbox refinement fails**: Need proper response-level blending
⏳ **Pending**: Response map blending or quality-based switching

### Key Insight
**OpenCV TrackerCSRT is already very good** (AUC 0.700). Adding deep features needs careful integration:
- Bbox-level refinement doesn't help (loses spatial precision)
- Response-level blending may work better (preserve peak location)
- Quality-based switching could be simpler (fallback mechanism)
