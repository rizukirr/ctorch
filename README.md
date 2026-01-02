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
- [x] **Affine Transform** (Linear/Dense layer) - Matrix multiplication with bias
- [ ] Convolutional layers
- [ ] Recurrent layers (RNN, LSTM, GRU)

#### Activation Functions
- [x] **ReLU** - Rectified Linear Unit
- [x] **Sigmoid** - Logistic activation
- [x] **Softmax** - Normalized exponential for classification
- [x] **Tanh** - Hyperbolic tangent activation
- [ ] Leaky ReLU
- [ ] ELU

#### Loss Functions
- [x] **Cross Entropy** - Numerically stable implementation with log-sum-exp trick
- [ ] Mean Squared Error (MSE)
- [ ] Mean Absolute Error (MAE)
- [ ] Binary Cross Entropy

### Training
- [x] **Backpropagation** - Automatic gradient computation for all layers and activations
- [x] **Optimizers**
  - [x] Stochastic Gradient Descent (SGD)
  - [ ] Adam
  - [ ] RMSprop
  - [ ] Momentum
- [ ] Learning rate schedulers
- [ ] Gradient clipping

### Advanced Features
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] Convolution operations
- [ ] Pooling layers (Max, Average)
- [ ] Broadcasting operations
- [x] **Element-wise operations** - Multiplication, subtraction
- [ ] Model save/load (serialization)
- [ ] GPU acceleration (CUDA/OpenCL)

## Build

### Quick Build (Recommended)
```bash
# Debug build (with AddressSanitizer and debug symbols)
./build.sh

# Release build (optimized with -O3)
./build.sh release

# Build and run with timing
./run.sh build
```

### Manual Build
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

The debug build includes AddressSanitizer (`-fsanitize=address`) for detecting memory errors.

## Usage Example

### Low-Level API
```c
#include "tensor.h"
#include "ops.h"

int main(void) {
    // Create tensor context for memory management (with gradient tracking)
    TensorContext *ctx = tensor_create(true);

    // Create input data (100 samples x 2 features)
    Tensor *inputs = tensor_randn(ctx, 100, 2);

    // Create weights (2 input features x 4 neurons)
    Tensor *weights = tensor_randn(ctx, 2, 4);

    // Create bias
    float *bias = scalar_randn(ctx, 4);

    // Forward pass: affine transform
    Tensor *logits = affine_transform(ctx, inputs, weights, bias);

    // Apply activation function
    softmax(logits);

    // Print results
    tensor_print(logits, 10, false);

    // Cleanup
    tensor_free(ctx);
    return 0;
}
```

### High-Level API (Keras-style)
```c
#include "keras.h"

int main(void) {
    // Initialize dense layer context
    DenseContext *ctx = dense_init(2);  // 2 input features

    // Create layers
    Dense *layer1 = dense_create(ctx, 4);  // 4 neurons
    Dense *layer2 = dense_create(ctx, 3);  // 3 output classes

    // Create input
    TensorContext *tensor_ctx = tensor_create(true);
    Tensor *inputs = tensor_randn(tensor_ctx, 100, 2);

    // Forward pass
    Tensor *out1 = dense_forward(ctx, layer1, inputs, ReLU);
    Tensor *out2 = dense_forward(ctx, layer2, out1, Softmax);

    // Cleanup
    dense_free(ctx);
    tensor_free(tensor_ctx);
    return 0;
}
```

See `main.c` for a complete example with spiral dataset generation.

## API Documentation

All public functions include comprehensive docstrings in the header files:
- `include/tensor.h` - Tensor creation, manipulation, and memory management
- `include/ops.h` - Neural network operations (affine, activations, loss, backprop)
- `include/keras.h` - High-level Keras-style API for building models
- `include/arena.h` - Custom arena allocator for efficient memory management
- `include/randn.h` - Random number generation utilities

## Current Limitations

- Limited to CPU operations (no GPU acceleration)
- Basic optimizer (only SGD, no Adam/RMSprop)
- No model persistence (serialization)
- No convolutional or recurrent layers
- No batch normalization or dropout

## Recent Updates

### v0.0.0-dev02 - Backpropagation & Training Support
- **Implemented full backpropagation** - Gradient computation for all layers and activations
- Added backward passes for:
  - Affine transformation (linear layer)
  - ReLU, Sigmoid, Tanh activations
  - Combined Softmax + Cross-Entropy loss
- **Implemented SGD optimizer** - Stochastic Gradient Descent with configurable learning rate
- Added gradient tracking with `requires_grad` parameter in `tensor_create()`
- Enhanced Dense layer API with `dense_backward()` and `dense_zero_grad()`
- Added utility functions: `tensor_zeros()`, `tensor_subtract()`, `tensor_accuracy()`
- **Documentation overhaul** - Added/fixed docstrings for all public API functions
- Added `scalar_randn()` for 1D random array generation

### v0.0.0-dev01 - Major API Refactoring
- Renamed `Vector` to `Tensor` throughout codebase for better semantic clarity
- Added comprehensive error handling system with `errors.h`
- Implemented Keras-style high-level API (`keras.h`)
- Reorganized codebase: split monolithic `ctorch.h` into modular headers (`ops.h`, `tensor.h`, `keras.h`)
- Added **Tanh** activation function
- Improved cross-entropy with numerically stable log-sum-exp implementation
- Fixed documentation mismatches and parameter typos
- Updated all function signatures and examples to use new Tensor API

### Breaking Changes (v0.0.0-dev01)
- All `vector_*` functions renamed to `tensor_*`
- `VectorContext` renamed to `TensorContext`
- Activation functions simplified: `activation_ReLU()` to `relu()`, `activation_softmax()` to `softmax()`
- Error handling: Functions now use global error context instead of return codes

### Breaking Changes (v0.0.0-dev02)
- Float precision used throughout (changed from `double` to `float`)

## Roadmap

1. ✅ Implement cross entropy loss function
2. ✅ Add tanh activation function
3. ✅ Add backpropagation for gradient computation
4. ✅ Implement SGD optimizer
5. 🚧 Build a complete training loop example (in progress)
6. Add more activation functions (leaky ReLU, ELU)
7. Add more loss functions (MSE, MAE, Binary Cross Entropy)
8. Implement advanced optimizers (Adam, RMSprop, Momentum)
9. Add convolutional layers
10. Implement batch normalization and dropout
11. Model serialization (save/load weights)
12. Performance optimization (SIMD, parallel computation)

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
