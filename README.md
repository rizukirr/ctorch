# ctorch

A toy PyTorch reimplementation in pure C for building simple neural networks from scratch.

## Overview

ctorch is a minimal deep learning library written in C that implements core neural network functionality. This is a hobby project to explore how neural networks work at a low level, without the abstractions of modern frameworks.

## Features

### Core Infrastructure
- [x] **Memory Management** - Custom arena allocator for efficient memory handling
- [x] **Vector/Matrix Operations** - 2D tensor-like data structure with dynamic sizing
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
- [ ] Tanh
- [ ] Leaky ReLU
- [ ] ELU

#### Loss Functions
- [ ] **Cross Entropy** (declared, not implemented)
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
#include "vector.h"

int main(void) {
    // Create vector context for memory management
    VectorContext *ctx = vector_create();

    // Create input data (100 samples x 2 features)
    Vector *inputs = vector_randn(ctx, 100, 2);

    // Create weights (4 neurons x 2 input features)
    Vector *weights = vector_randn(ctx, 4, 2);

    // Create bias
    double bias[4] = {0.0, 0.0, 0.0, 0.0};

    // Forward pass: affine transform
    Vector *logits = affine_transform(ctx, inputs, weights, bias);

    // Apply activation function
    activation_softmax(logits);

    // Print results
    vector_print(logits, 10, true);

    // Cleanup
    vector_free(ctx);
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

## Roadmap

1. Implement cross entropy loss function
2. Add backpropagation for gradient computation
3. Implement SGD optimizer
4. Add more activation functions (tanh, leaky ReLU)
5. Build a complete training loop example
6. Add convolutional layers
7. Implement batch normalization and dropout
8. Model serialization

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
