#ifndef CTORCH_KERAS_H
#define CTORCH_KERAS_H

#include "tensor.h"

typedef struct DenseContext DenseContext;

typedef enum { None, ReLU, Sigmoid, Softmax, Tanh } Activation;

typedef struct Dense {
  Tensor *weights;
  float *biases;
  float *grad_biases;     // Gradient for biases
  Tensor *outputs;        // Cached output (post-activation)
  Tensor *inputs;         // Cached input from forward pass (for backward)
  Tensor *pre_activation; // Cached pre-activation (for backward through
                          // activations)
  Activation activation;  // Activation function used
  size_t output_size;     // Number of output neurons (for gradient allocation)
} Dense;

typedef enum { CrossEntropy } Loss;

/**
 * @brief Initializes a new Dense layer context.
 *
 * Creates a context for managing Dense layer allocations using arena allocator.
 * The input size is tracked so that subsequent layers can be created with
 * compatible dimensions.
 *
 * @param input_size Number of input features for the first layer in this
 * context
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
 * @brief Performs backward pass through a Dense layer.
 *
 * Computes gradients with respect to inputs, weights, and biases.
 * Gradients are accumulated in the weight and bias tensors.
 * Returns gradient with respect to inputs for further backpropagation.
 *
 * @param ctx Dense layer context
 * @param dense The Dense layer to backpropagate through
 * @param grad_output Gradient flowing back from the next layer
 * @return Gradient with respect to inputs, or NULL on error
 */
Tensor *dense_backward(DenseContext *ctx, Dense *dense, Tensor *grad_output);

/**
 * @brief Zeros all gradients in the Dense layer.
 *
 * Resets weight and bias gradients to zero in preparation for a new
 * backward pass.
 *
 * @param dense The Dense layer whose gradients should be zeroed
 */
void dense_zero_grad(Dense *dense);

/**
 * @brief Performs SGD optimization step for all Dense layers.
 *
 * Updates weights and biases using stochastic gradient descent:
 * weights = weights - learning_rate * grad_weights
 * biases = biases - learning_rate * grad_biases
 *
 * @param ctx Dense layer context
 * @param layers Array of Dense layer pointers
 * @param num_layers Number of layers in the array
 * @param learning_rate Learning rate (step size) for optimization
 * @return 0 on success, negative error code on failure
 */
int sgd_step(DenseContext *ctx, Dense **layers, size_t num_layers,
             float learning_rate);

/**
 * @brief Frees a Dense layer context and all associated layers.
 *
 * Releases all memory allocated for the Dense context, including all layers
 * created with this context.
 *
 * @param ctx DenseContext to free
 * @return 0 on success, negative error code on failure
 */
int dense_free(DenseContext *ctx);

#endif // CTORCH_KERAS_H
