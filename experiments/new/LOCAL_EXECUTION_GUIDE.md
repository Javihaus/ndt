# Running Experiments on Your Local Machine
## With Your Downloaded Datasets

---

## 🚨 **IMPORTANT: Cloud vs Local Environment**

**Current Situation:**
- ✅ You have datasets on YOUR local computer in `experiments/new/data/`
- ❌ This cloud environment (where I run) **CANNOT access** your local files
- ✅ I've created all the code needed to run experiments
- 🎯 **You need to run experiments on YOUR machine**

---

## 📁 **Verify Your Setup**

Make sure you have this structure on YOUR local computer:

```
your-ndt-repo/
└── experiments/new/
    ├── data/               # Your datasets
    │   ├── cifar-10/
    │   │   └── cifar-10-batches-py/
    │   │       ├── data_batch_1
    │   │       ├── data_batch_2
    │   │       ├── ...
    │   │       └── test_batch
    │   ├── imagenet/
    │   │   └── train-00001-of-00021.parquet
    │   ├── glue/
    │   │   └── mnli/
    │   │       └── train-00000-of-00001.parquet
    │   └── conceptual_captions/
    │       └── [your files]
    │
    ├── modern_dataset_loaders.py     # ← I created this
    ├── run_experiments_parallel.py   # ← I created this
    └── phase1_calibration.py         # ← Updated by me
```

---

## 🚀 **Running Experiments Locally (Parallel)**

### **Option 1: Parallel Execution (FAST - ~10-20 hours)**

```bash
cd /path/to/your/ndt/experiments/new

# Run 4 experiments in parallel (recommended)
python run_experiments_parallel.py --num-processes 4 --num-steps 2000

# Or more aggressive (if you have 8+ CPU cores)
python run_experiments_parallel.py --num-processes 8 --num-steps 2000
```

**Benefits:**
- ✅ 4x faster than sequential
- ✅ ~10-20 hours for all 30 experiments
- ✅ Uses multiple CPU cores

**Requirements:**
- At least 4 CPU cores
- 8-16GB RAM

---

### **Option 2: Sequential Execution (SLOW - ~40-80 hours)**

```bash
cd /path/to/your/ndt/experiments/new

# Run one at a time
python phase1_calibration.py --num-steps 2000 --output-dir results/phase1_full
```

---

## 📊 **What Will Happen**

### **30 Real Experiments Will Run:**

| Architecture Type | Dataset | Count |
|------------------|---------|-------|
| MLPs (8 variants) | MNIST + CIFAR-10 | 16 |
| CNNs (3 variants) | CIFAR-10 | 3 |
| ResNet18 | CIFAR-10 + ImageNet | 2 |
| Transformers (5 variants) | GLUE + CIFAR-10 | 9 |
| **TOTAL** | | **30** |

### **Each Experiment:**
- 2000 training steps
- 400 dimensionality measurements (every 5 steps)
- Real gradient descent on real data
- Saves ~500KB-1.5MB JSON result

---

## ⏱️ **Time Estimates**

| Method | Time | Notes |
|--------|------|-------|
| **Parallel (4 cores)** | 10-20 hours | **Recommended** |
| **Parallel (8 cores)** | 5-10 hours | If you have powerful CPU |
| Sequential | 40-80 hours | Not recommended |

**Per experiment:** ~30-60 minutes

---

## 🔍 **Monitoring Progress**

While running, you'll see:

```
======================================================================
PARALLEL EXPERIMENT RUNNER
======================================================================
Total experiments: 30
Parallel processes: 4
...

[1/30] ✓ Completed mlp_shallow_2 × cifar10 in 45.2 min
[2/30] ⏭  Skipped mlp_shallow_2 × mnist (already exists)
[3/30] ✓ Completed cnn_shallow × cifar10 in 52.1 min
...
```

**Results saved to:** `experiments/new/results/phase1_full/`

---

## 🐛 **If Something Fails**

### **Dataset Not Found:**
```
✗ Failed mlp_shallow_2 × cifar10: [Errno 2] No such file or directory
```

**Fix:** Check that `experiments/new/data/cifar-10/cifar-10-batches-py/data_batch_1` exists

### **Out of Memory:**
```
RuntimeError: CUDA out of memory
```

**Fix:** Reduce number of parallel processes:
```bash
python run_experiments_parallel.py --num-processes 2
```

### **Crashes:**
The runner is designed to **resume from where it left off**. Just run the command again - it will skip completed experiments.

---

## 💾 **After Completion**

### **You'll Have:**
- ✅ 30 JSON result files (~20-40MB total)
- ✅ Real training data for TAP validation
- ✅ Ready for Phase 2-4 analysis

### **Next Steps:**
1. Commit results to git:
```bash
git add experiments/new/results/phase1_full/*.json
git commit -m "Add Phase 1 results: 30 experiments with modern datasets"
git push origin claude/review-repo-cla-01QN2r8oCF5Ao3Z1BGhtCgSE
```

2. Run Phase 1 analysis:
```bash
python phase1_analysis.py --results-dir results/phase1_full
```

---

## ❓ **Questions?**

- **Can I run just a subset?** Edit `EXPERIMENT_PLAN` in `run_experiments_parallel.py`
- **Can I use GPU?** Yes, it will auto-detect and use if available
- **Can I stop/resume?** Yes, it skips completed experiments automatically

---

## 🎯 **Start Now**

```bash
cd /path/to/your/ndt/experiments/new
python run_experiments_parallel.py --num-processes 4 --num-steps 2000
```

**Let it run overnight!** ☕
