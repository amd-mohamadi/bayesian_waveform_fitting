### **The Core Strategy: "Stretch & Align"**

**The Concept:**
Instead of trying to teach a Neural Network to ignore the time/frequency difference (Siamese), we will use **Soft-DTW** to mathematically *un-stretch* the signals.

* **Real Data (High Freq):** 0.5s duration  Resampled to  points.
* **Synthetic Data (Low Freq):** 2.0s duration  Resampled to  points.
* **The Magic:** Soft-DTW calculates the "energy cost" to align the shapes of these two -point vectors. If the Moment Tensor is correct, the shapes will align perfectly (low cost), even though one represents 0.5s and the other 2.0s.

---

### **Step 1: The Preprocessing Pipeline (Crucial)**

*Goal: Ensure inputs to Soft-DTW are strictly shape-based, removing absolute amplitude and time constraints.*

1. **Select Windows:**
* **Observed:** Cut a tight window around the P/S arrivals (e.g., 0.5s).
* **Synthetic:** Cut a loose, longer window (e.g., 2.0s) to capture the full low-frequency waveform.


2. **Resample to Fixed Size ():**
* Force both arrays to the same length (e.g., `N=128` or `N=256`).
* *Why?* Soft-DTW is . Keeping  small makes the inversion lightning fast while preserving the "shape" info needed for the Moment Tensor.


3. **Z-Score Normalization (Per Trace):**
* Subtract Mean, Divide by Std Dev for *each* trace individually.
* *Why?* This forces the loss to focus purely on the **radiation pattern** (relative shape) and ignores absolute scaling errors.



---

### **Step 2: The Soft-DTW Implementation**

*Goal: Replace the L2/Siamese Loss with a Differentiable DTW Loss.*

You don't need to write the CUDA kernel yourself. Use the `soft-dtw-cuda` or a pure PyTorch implementation.

**The Equation:**


* ** (Gamma):** This is your "temperature" hyperparameter.
* : Behaves like hard DTW (finds the *exact* single best path). Good for final precision.
* : Behaves like a "blurred" alignment. Smoother gradients, better for the early stages of inversion to avoid local minima.



**Code Snippet (PyTorch):**

```python
import torch
import torch.nn as nn

def soft_dtw_loss(x, y, gamma=0.1):
    """
    Computes Soft-DTW loss between batch of waveforms.
    x: (Batch, Length) - Synthetic (Axitra)
    y: (Batch, Length) - Observed (Real)
    gamma: Smoothing parameter (start with 0.1)
    """
    # Calculate pairwise distance matrix (Euclidean or Cosine)
    # For waveforms, simple Euclidean distance between points is standard
    # D[i,j] = (x[i] - y[j])^2
    
    # Note: For optimized implementation, use the 'soft_dtw_cuda' library
    # pip install soft-dtw-cuda
    from soft_dtw_cuda import SoftDTW 
    
    # Initialize the criterion once (put this outside your loop)
    criterion = SoftDTW(use_cuda=True, gamma=gamma)
    
    # Compute loss
    loss = criterion(x, y)
    
    return loss.mean()

```

---

### **Step 3: Integration into Inversion Loop**

*Goal: Swap the objective function.*

**Current (L2 / Siamese):**

```python
# Old Way
diff = synthetic - observed
loss = torch.sum(diff ** 2) # L2
# OR
loss = siamese_net(synthetic, observed) # Siamese

```

**New (Soft-DTW):**

```python
# New Way
# 1. Resample Step (if not done in preprocessing)
# synthetic_resampled = torch.nn.functional.interpolate(synthetic, size=128)
# observed_resampled = torch.nn.functional.interpolate(observed, size=128)

# 2. Compute Loss
loss = soft_dtw_loss(synthetic_resampled, observed_resampled, gamma=0.1)

# 3. Backprop (if doing gradient-based inversion)
loss.backward() 

```

---

### **Step 4: The "Sanity Check" (Visualization)**

*Goal: Verify that Soft-DTW is actually aligning the "Stretched" features.*

Before running the full inversion, run this test on **one** event:

1. Take your 0.5s Real waveform (High Freq).
2. Take your 2.0s Synthetic waveform (Low Freq).
3. Compute the **Alignment Path** (often called the "Transport Plan" or "Warping Path").
4. Plot the matrix. You should see a diagonal line (roughly) that "wiggles" to match the peaks.

* **If the path is a straight diagonal:** It means no warping is needed (unlikely for your data).
* **If the path breaks/jumps:** Your  might be too low, or the polarity is flipped (which is good! High cost = wrong MT).
* **If the path smoothly curves:** It works! It is successfully mapping the "slow" synthetic peak to the "fast" observed peak.
