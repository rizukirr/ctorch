# ctorch

A toy PyTorch reimplementation in pure C for building simple neural networks from scratch.

## Overview

ctorch is a minimal deep learning library written in C that implements core neural network functionality. This is a hobby project to explore how neural networks work at a low level, without the abstractions of modern frameworks.

## Features

### Core Infrastructure
- [x] **Memory Management** - Custom arena allocator for efficient memory handling
- [x] **Tensor/Matrix Operations** - 2D tensor-like data structure with dynamic sizing
- [x] **Random Number Generation** - Box-Muller transform for Gaussian random initialization

### Neural Network Components

#### Layers
- [x] **Linear** (Dense/Fully-connected layer) - `linear(ctx, input, weight, bias)` - Matrix multiplication with bias, like PyTorch's `nn.Linear`
- [ ] Convolutional layers
- [ ] Recurrent layers (RNN, LSTM, GRU)

#### Activation Functions
- [x] **ReLU** - `relu(input)` - Rectified Linear Unit
- [x] **Sigmoid** - `sigmoid(input)` - Logistic activation
- [x] **Softmax** - `softmax(input)` - Normalized exponential for classification
- [x] **Tanh** - `tanh_(input)` - Hyperbolic tangent activation (underscore to avoid math.h conflict)
- [ ] Leaky ReLU
- [ ] ELU

#### Loss Functions
- [x] **Cross Entropy** - `cross_entropy(ctx, logits, target)` - Numerically stable implementation with log-sum-exp trick
- [x] **MSE Loss** - `mse_loss(ctx, prediction, target)` - Element-wise squared error with 0.5 factor for cleaner gradients
- [ ] Mean Absolute Error (MAE)
- [ ] Binary Cross Entropy

### Training
- [x] **Backpropagation** - Full gradient computation via reverse-mode autodiff
- [x] **Optimizers** - Adam, SGD with momentum support via `dense_backward(ctx, grad_output, lr, optimizer_type)`
- [ ] Separate optimizer objects with state persistence
- [ ] Learning rate schedulers
- [ ] Gradient clipping

### Advanced Features (TODO)
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] Convolution operations
- [ ] Pooling layers (Max, Average)
- [ ] Broadcasting operations
- [ ] Element-wise operations
- [ ] Model save/load (serialization)
- [ ] GPU acceleration (CUDA/OpenCL)

## Build

```bash
# Debug build (includes AddressSanitizer)
./build.sh && cmake --build build

# Release build (optimized with -O3)
./build.sh release && cmake --build build

# Build and run
./run.sh build

# Run the executable
./build/bin/ctorch
```

## Usage Example

The repository includes a complete example in `main.c` that demonstrates training a 3-layer neural network on various synthetic datasets.

### Quick Start

```c
#include "keras.h"
#include "ops.h"
#include "tensor.h"

int main(void) {
    // Create data context for inputs/labels
    TensorContext *ctx = tensor_init();

    // Generate synthetic dataset (300 samples, 3 classes)
    Tensor *data = blob_data(ctx, 100, 3);  // Easy: Gaussian clusters
    Tensor *x = tensor_drop(ctx, data, 2, AxisColumn);   // Features
    Tensor *y = tensor_select(ctx, data, 2, AxisColumn); // Labels

    // Create network: 2 input features -> 4 -> 4 -> 4 -> 3 classes
    DenseContext *model = dense_init(x->cols);
    dense_create(model, 4, ReLU);
    dense_create(model, 4, ReLU);
    dense_create(model, 4, ReLU);
    dense_create(model, 3, Linear);  // Output layer (raw logits)

    // Training loop
    for (size_t i = 0; i < 1000; i++) {
        Tensor *output = dense_forward(model, x);
        Tensor *softmax_output = softmax_dup(ctx, output);
        
        Tensor *loss = cross_entropy(ctx, output, y);
        Tensor *grad_output = cross_entropy_backward(ctx, softmax_output, y);
        
        dense_backward(model, grad_output, 0.01f, Adam);
        
        if (i % 100 == 0) {
            Tensor *preds = predict(model, x);
            float acc = accuracy(model, preds, y);
            float loss_avg = tensor_avg(loss);
            printf("Iter %zu: loss=%.6f, accuracy=%.2f%%\n", 
                   i, loss_avg, acc * 100);
        }
    }

    // Cleanup
    dense_free(model);
    tensor_free(ctx);
    return 0;
}
```

### Available Datasets

The example includes 4 synthetic datasets with varying difficulty:

| Dataset | Difficulty | Description | Expected Accuracy |
|---------|-----------|-------------|-------------------|
| `blob_data()` | **Easy** | Linearly separable Gaussian clusters | 100% |
| `circles_data()` | **Medium** | Concentric rings (requires nonlinear boundaries) | 100% |
| `xor_data()` | **Medium** | Quadrant-based XOR-style separation | 100% |
| `spiral_data()` | **Hard** | Interleaved spirals (complex boundaries) | ~44% |

**Example Results** (1000 iterations, Adam optimizer, lr=0.01):

```
=== Testing blob dataset ===
Iter    0: loss=1.468367, accuracy=2.33%
Iter  200: loss=0.037076, accuracy=100.00%
Iter  999: loss=0.002776, accuracy=100.00%

=== Testing spiral dataset ===
Iter    0: loss=1.215238, accuracy=35.00%
Iter  200: loss=1.077476, accuracy=42.00%
Iter  999: loss=1.005064, accuracy=44.00%
```

Change the dataset in `main.c` by switching between:
- `blob_data(ctx, 100, 3)`
- `circles_data(ctx, 100, 3)`
- `xor_data(ctx, 100, 3)`
- `spiral_data(ctx, 100, 3)`

## Current Limitations

- No separate optimizer objects (weight updates inline in backward pass, but supports Adam/SGD/Momentum)
- Limited to CPU operations (no BLAS, no GPU)
- No model persistence (save/load)
- No batch normalization or dropout
- 2D tensors only (no N-dimensional arrays)
- No dynamic computation graphs

## Recent Updates

### v0.0.0-dev04 - PyTorch-aligned Naming Convertion, Performance Optimizations & Bug Fixes (2025-01-28)
- **Critical Bug Fix**: Fixed uninitialized memory in `cross_entropy()` causing extremely high loss values
- **Performance**: Optimized hot paths with direct array access instead of `tensor_get()`
- **Performance**: Removed unnecessary `memset()` calls and temporary tensor allocations
- **Performance**: In-place activation functions (relu, sigmoid, tanh, softmax) - no memory allocation
- **Enhancement**: `cross_entropy_backward()` now auto-detects class indices vs one-hot encoding
- **Enhancement**: `mse_loss_backward()` supports both single-value and multi-value targets
- **Optimization**: Eliminated redundant `exp()` calls in softmax (2x speedup)
- **Optimization**: Replaced `powf(x, 2)` with `x * x` for better performance
- **Optimization**: Improved matrix multiplication loop order for cache locality
- **New**: Added 4 synthetic datasets: blobs (easy), circles (medium), xor (medium), spiral (hard)
- **Estimated speedup**: 3-7x faster training overall
- Renamed `affine_transform` to `linear` (like PyTorch's `nn.Linear`)
- Renamed `squared_error` to `mse_loss` and `squared_error_backward` to `mse_loss_backward`
- Renamed `tanhh` to `tanh_` (underscore avoids conflict with math.h)
- Standardized parameter names to match PyTorch conventions:
  - `inputs` → `input`, `weights` → `weight`, `biases` → `bias`
  - `upstream_grad`/`loss_grad` → `grad_output`
  - `y_true` → `target`, `y_pred` → `prediction`
  - `input_size`/`output_size` → `in_features`/`out_features`
  - `learning_rate` → `lr`
- Updated `Dense` struct: `weights`/`biases` → `weight`/`bias`

### v0.0.0-dev03 - Backpropagation & Training
- Implemented **backpropagation** via `dense_backward()` with gradient computation
- Added **MSE Loss** function with backward pass
- Implemented **backward pass operations**: `cross_entropy_backward`, `mse_loss_backward`, `relu_backward`, `sigmoid_backward`, `tanh_backward`
- Added gradient computation for linear layers: `weight_gradient`, `bias_gradient`, `input_gradient`
- Implemented `predict()` and `accuracy()` functions for model evaluation
- Basic SGD-style weight updates integrated into backward pass

### v0.0.0-dev02 - Major API Refactoring
- Renamed `Vector` to `Tensor` throughout codebase for better semantic clarity
- Added comprehensive error handling system with `errors.h`
- Implemented Keras-style high-level API (`keras.h`)
- Reorganized codebase: split monolithic `ctorch.h` into modular headers (`ops.h`, `tensor.h`, `keras.h`)
- Added **Tanh** activation function
- Improved cross-entropy with numerically stable log-sum-exp implementation

### Breaking Changes (v0.0.0-dev02)
- All `vector_*` functions renamed to `tensor_*`
- `VectorContext` renamed to `TensorContext`
- Activation functions simplified: `activation_ReLU()` to `relu()`, `activation_softmax()` to `softmax()`
- Error handling: Functions now use global error context instead of return codes

## Roadmap

1. [x] Implement cross entropy loss function
2. [x] Add tanh activation function
3. [x] Add backpropagation for gradient computation
4. [x] Implement basic SGD weight updates
5. [x] Build a complete training loop example
6. [x] Add Adam optimizer support
7. [x] Performance optimizations (direct array access, eliminate temp allocations)
8. [x] Multiple synthetic datasets for testing
9. [ ] Add separate optimizer objects with state persistence
10. [ ] Add more activation functions (Leaky ReLU, ELU, GELU)
11. [ ] Add convolutional layers
12. [ ] Implement batch normalization and dropout
13. [ ] Model serialization (save/load)
14. [ ] BLAS integration for faster matrix operations

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

This is a hobby project. Feel free to fork and experiment!
