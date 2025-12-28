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

### Training (TODO)
- [ ] **Backpropagation** - Automatic gradient computation
- [ ] **Optimizers**
  - [ ] Stochastic Gradient Descent (SGD)
  - [ ] Adam
  - [ ] RMSprop
  - [ ] Momentum
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
mkdir build && cd build
cmake ..
make
```

## Usage Example

```c
#include "ctorch.h"
#include "tensor.h"

int main(void) {
    // Create tensor context for memory management
    TensorContext *ctx = tensor_create();

    // Create input data (100 samples x 2 features)
    Tensor *inputs = tensor_randn(ctx, 100, 2);

    // Create weights (4 neurons x 2 input features)
    Tensor *weights = tensor_randn(ctx, 4, 2);

    // Create bias
    double bias[4] = {0.0, 0.0, 0.0, 0.0};

    // Forward pass: affine transform
    Tensor *logits = affine_transform(ctx, inputs, weights, bias);

    // Apply activation function
    softmax(logits);

    // Print results
    tensor_print(logits, 10, true);

    // Cleanup
    tensor_free(ctx);
    return 0;
}
```

See `main.c` for a complete example with spiral dataset generation.

## Current Limitations

- No backpropagation (forward pass only)
- No gradient computation or training
- Limited to CPU operations
- Basic matrix operations only
- No model persistence

## Recent Updates

### v0.2.0 - Major API Refactoring
- Renamed `Vector` to `Tensor` throughout codebase for better semantic clarity
- Added comprehensive error handling system with `errors.h`
- Implemented Keras-style high-level API (`keras.h`)
- Reorganized codebase: split monolithic `ctorch.h` into modular headers (`ops.h`, `tensor.h`, `keras.h`)
- Added **Tanh** activation function
- Improved cross-entropy with numerically stable log-sum-exp implementation
- Fixed documentation mismatches and parameter typos
- Updated all function signatures and examples to use new Tensor API

### Breaking Changes
- All `vector_*` functions renamed to `tensor_*`
- `VectorContext` renamed to `TensorContext`
- Activation functions simplified: `activation_ReLU()` to `relu()`, `activation_softmax()` to `softmax()`
- Error handling: Functions now use global error context instead of return codes

## Roadmap

1. [DONE] Implement cross entropy loss function
2. [DONE] Add tanh activation function
3. Add backpropagation for gradient computation
4. Implement SGD optimizer
5. Add more activation functions (leaky ReLU, ELU)
6. Build a complete training loop example
7. Add convolutional layers
8. Implement batch normalization and dropout
9. Model serialization

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
