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
- [x] **Squared Error** - Element-wise squared error with 0.5 factor for cleaner gradients
- [ ] Mean Absolute Error (MAE)
- [ ] Binary Cross Entropy

### Training
- [x] **Backpropagation** - Gradient computation via `dense_backward()`
- [x] **Basic SGD** - Inline weight updates in backward pass
- [ ] **Separate Optimizer Objects**
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

```c
#include "keras.h"
#include "ops.h"
#include "tensor.h"

int main(void) {
    // Create data context for inputs/labels
    TensorContext *data_ctx = tensor_create();

    // Input data: 4 samples, 2 features
    Tensor *X = tensor_new(data_ctx, 2);
    float x_data[4][2] = {{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}, {0.7f, 0.8f}};
    for (int i = 0; i < 4; i++)
        tensor_append(data_ctx, X, x_data[i]);

    // Labels: one-hot encoded (4 samples, 3 classes)
    Tensor *Y = tensor_new(data_ctx, 3);
    float y_data[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}};
    for (int i = 0; i < 4; i++)
        tensor_append(data_ctx, Y, y_data[i]);

    // Create network: 2 -> 4 (ReLU) -> 3 (Softmax)
    DenseContext *model = dense_init(2);
    dense_create(model, 4, ReLU);
    dense_create(model, 3, Softmax);

    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        Tensor *output = dense_forward(model, X);
        Tensor *loss_grad = cross_entropy_backward(data_ctx, output, Y);
        dense_backward(model, 0.5f, loss_grad);
    }

    // Evaluate
    Tensor *predictions = predict(model, X);
    float acc = accuracy(model, predictions, Y);
    printf("Accuracy: %.3f\n", acc);

    // Cleanup
    dense_free(model);
    tensor_free(data_ctx);
    return 0;
}
```

See `main.c` for a complete example.

## Current Limitations

- No separate optimizer objects (weight updates inline in backward pass)
- Limited to CPU operations
- Basic matrix operations only (no BLAS)
- No model persistence (save/load)
- No batch normalization or dropout

## Recent Updates

### v0.3.0 - Backpropagation & Training
- Implemented **backpropagation** via `dense_backward()` with gradient computation
- Added **Squared Error** loss function with backward pass
- Implemented **backward pass operations**: `cross_entropy_backward`, `squared_error_backward`, `relu_backward`, `sigmoid_backward`, `tanh_backward`
- Added gradient computation for affine layers: `weight_gradient`, `bias_gradient`, `input_gradient`
- Implemented `predict()` and `accuracy()` functions for model evaluation
- Basic SGD-style weight updates integrated into backward pass

### v0.2.0 - Major API Refactoring
- Renamed `Vector` to `Tensor` throughout codebase for better semantic clarity
- Added comprehensive error handling system with `errors.h`
- Implemented Keras-style high-level API (`keras.h`)
- Reorganized codebase: split monolithic `ctorch.h` into modular headers (`ops.h`, `tensor.h`, `keras.h`)
- Added **Tanh** activation function
- Improved cross-entropy with numerically stable log-sum-exp implementation

### Breaking Changes (v0.2.0)
- All `vector_*` functions renamed to `tensor_*`
- `VectorContext` renamed to `TensorContext`
- Activation functions simplified: `activation_ReLU()` to `relu()`, `activation_softmax()` to `softmax()`
- Error handling: Functions now use global error context instead of return codes

## Roadmap

1. [DONE] Implement cross entropy loss function
2. [DONE] Add tanh activation function
3. [DONE] Add backpropagation for gradient computation
4. [DONE] Implement basic SGD weight updates
5. [DONE] Build a complete training loop example
6. Add separate optimizer objects (Adam, RMSprop, Momentum)
7. Add more activation functions (Leaky ReLU, ELU)
8. Add convolutional layers
9. Implement batch normalization and dropout
10. Model serialization (save/load)

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
