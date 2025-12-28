#ifndef CTORCH_KERAS_H
#define CTORCH_KERAS_H

#include "tensor.h"

typedef struct DenseContext DenseContext;

typedef struct {
  Tensor *weights;
  Tensor *biases;
  Tensor *result;
} Dense;

typedef enum { ReLU, Sigmoid, Softmax, Tanh } Activation;

typedef enum { CrossEntropy } Loss;

/**
 * @brief Initializes a new Dense layer context.
 *
 * Creates a context for managing Dense layer allocations using arena allocator.
 * The input size is tracked so that subsequent layers can be created with
 * compatible dimensions.
 *
 * @param input_size Number of input features for the first layer in this context
 * @return Pointer to new DenseContext, or NULL on allocation failure
 */
DenseContext *dense_init(size_t input_size);

/**
 * @brief Creates a new Dense (fully connected) layer.
 *
 * Allocates and initializes a dense layer with random weights and biases.
 * The input size is determined from the context (set by dense_init() or
 * updated by previous layer creation).
 *
 * @param ctx Dense layer context for allocation
 * @param output_size Number of output features (neurons) in this layer
 * @return Pointer to new Dense layer, or NULL on error
 */
Dense *dense_create(DenseContext *ctx, size_t output_size);

/**
 * @brief Performs forward pass through a Dense layer.
 *
 * Computes the output of the dense layer: output = inputs * weights + biases
 *
 * @param ctx Dense layer context
 * @param dense The Dense layer to use
 * @param inputs Input tensor
 * @param activation Activation function to use
 * @return Output tensor, or NULL on error
 */
Tensor *dense_forward(DenseContext *ctx, Dense *dense, Tensor *inputs,
                      Activation activation);

/**
 * @brief Frees a Dense layer context and all associated layers.
 *
 * Releases all memory allocated for the Dense context, including all layers
 * created with this context.
 *
 * @param ctx DenseContext to free
 */
void dense_free(DenseContext *ctx);

#endif // CTORCH_KERAS_H
