# Google Colab Execution Instructions

## Quick Start

1. **Upload the notebook to Google Colab:**
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Select: `Checkpoint_Experiments_Colab.ipynb`

2. **Set GPU runtime:**
   - Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: A100 (or V100 if A100 not available)
   - Click Save

3. **Run all cells:**
   - Runtime → Run all
   - Or: Press Shift+Enter on each cell sequentially

4. **Wait for completion:**
   - Expected time: ~90 minutes on A100
   - Expected time: ~2-3 hours on V100

5. **Download results:**
   - Last cell will automatically download `checkpoint_experiments_results.zip`
   - Or manually: Files tab (left) → Right-click zip → Download

## What Gets Generated

### Checkpoints (9 total, ~75MB)
```
checkpoints/
├── transformer_deep_mnist/
│   ├── checkpoint_step_00100.pt
│   ├── checkpoint_step_01000.pt
│   └── checkpoint_step_02000.pt
├── cnn_deep_mnist/
│   ├── checkpoint_step_00100.pt
│   ├── checkpoint_step_01000.pt
│   └── checkpoint_step_02000.pt
└── mlp_narrow_mnist/
    ├── checkpoint_step_00100.pt
    ├── checkpoint_step_01000.pt
    └── checkpoint_step_02000.pt
```

### Results (4 JSON files)
```
results/
├── transformer_deep_mnist_result.json
├── cnn_deep_mnist_result.json
├── mlp_narrow_mnist_result.json
└── summary.json
```

## After Downloading

### Option 1: Provide zip file directly
Just share the downloaded `checkpoint_experiments_results.zip` file and I'll:
1. Extract it
2. Run the modified Phase 2 analysis
3. Generate final results and visualizations

### Option 2: Extract and provide files
If you prefer to extract first:
1. Unzip the file
2. Share the contents (checkpoints and results folders)

## Troubleshooting

### Out of Memory Error
- Reduce batch size in cell 4: Change `batch_size=64` to `batch_size=32`
- Or use V100 instead of A100 (more memory efficient for these small models)

### Download Failed
- Go to Files tab (folder icon on left)
- Right-click `checkpoint_experiments_results.zip`
- Select "Download"

### Execution Interrupted
- Click "Runtime → Run all" again
- Completed experiments will be skipped (checks for existing checkpoints)
- Only remaining experiments will run

### CUDA Out of Memory
- Restart runtime: Runtime → Restart runtime
- Change to V100: Runtime → Change runtime type → V100
- Run again

## Expected Output

### Console Output
```
Using device: cuda
GPU: NVIDIA A100-SXM4-40GB
Memory: 40.00 GB

Loading MNIST dataset...
✓ Data loaded: 60000 train, 10000 test

======================================================================
EXPERIMENT: transformer_deep_mnist
======================================================================
Model: transformer_deep
Parameters: 243,978

Training transformer_deep_mnist...
Target steps: 2000
Checkpoint steps: [100, 1000, 2000]
transformer_deep_mnist: 100%|██████████| 2000/2000 [05:23<00:00, 6.18it/s, loss=0.0234]

✓ Saved checkpoint: checkpoint_step_00100.pt
✓ Saved checkpoint: checkpoint_step_01000.pt
✓ Saved checkpoint: checkpoint_step_02000.pt

✓ Final accuracy: 0.9678
✓ Completed in 5.4 minutes

[... similar for cnn_deep_mnist and mlp_narrow_mnist ...]

======================================================================
ALL EXPERIMENTS COMPLETE
======================================================================
Total time: 92.3 minutes (1.5 hours)
Experiments: 3
Checkpoints saved: 9

Accuracies:
  transformer_deep_mnist: 0.9678
  cnn_deep_mnist: 0.9812
  mlp_narrow_mnist: 0.9523
```

## What Happens Next

Once you provide the results, I will:

1. **Load checkpoints** into modified_phase2_analysis.py
2. **Extract features:**
   - Transformers: Attention patterns
   - CNNs: Convolutional filters
   - MLPs: Hidden layer activations
3. **Compare early vs late:**
   - Cosine similarity matrices
   - Feature diversity measures
   - Qualitative visualizations
4. **Test hypothesis:** Are early features qualitatively different from late?
5. **Generate report** with honest framing

## Timeline

- **Your Colab execution:** ~90 minutes
- **My analysis:** ~15 minutes
- **Total:** ~2 hours from start to final results

## File Size

- **Zip file:** ~75MB
- **Extracted:** ~75MB (same, just uncompressed)
- **Upload time:** < 1 minute with good connection

---

**Ready!** Just run the notebook on Colab and send me the results when done.
