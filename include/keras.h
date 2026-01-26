#ifndef CTORCH_KERAS_H
#define CTORCH_KERAS_H

#include "tensor.h"

typedef enum { ReLU, Sigmoid, Softmax, Tanh } Activation;
typedef enum { CrossEntropy } Loss;

typedef struct DenseContext DenseContext;

typedef struct {
  Tensor *weights;
  Tensor *biases;
  struct Dense *next;
  Activation activation;
} Dense;

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
 * updated by previous layer creation). The layer is added to the context's
 * internal layer stack.
 *
 * @param ctx Dense layer context for allocation
 * @param output_size Number of output features (neurons) in this layer
 * @param activation Activation function to apply after this layer
 * @return 0 on success, negative CTorchError code on failure
 */
int dense_create(DenseContext *ctx, size_t output_size, Activation activation);

/**
 * @brief Performs forward pass through all Dense layers in the context.
 *
 * Runs the input through each layer in sequence, applying affine transformation
 * (output = inputs * weights + biases) followed by each layer's activation
 * function. The final output is cached in ctx->output.
 *
 * @param ctx Dense layer context containing the layer stack
 * @param inputs Input tensor of shape (batch_size, input_features)
 * @return Output tensor after all layers, or NULL on error
 */
Tensor *dense_forward(DenseContext *ctx, Tensor *inputs);

/**
 * @brief Frees a Dense layer context and all associated layers.
 *
 * Releases all memory allocated for the Dense context, including all layers
 * created with this context.
 *
 * @param ctx DenseContext to free
 */
void dense_free(DenseContext *ctx);

/**
 * @brief Performs backward pass (backpropagation) through all Dense layers.
 *
 * Computes gradients and updates weights/biases for all layers using the
 * cached intermediate values from the forward pass. Must be called after
 * dense_forward().
 *
 * @param ctx Dense layer context containing the layer stack and cached values
 * @param learning_rate Step size for gradient descent weight updates
 * @param loss_grad Gradient of loss with respect to network output (dL/dY)
 * @return 0 on success, negative CTorchError code on failure
 */
int dense_backward(DenseContext *ctx, float learning_rate, Tensor *loss_grad);

/**
 * @brief Computes predictions for the given input data.
 *
 * Computes the output of the network for the given input data. The output is
 * a one-hot encoded tensor with the predicted class for each sample.
 *
 * @param ctx Dense layer context containing the layer stack
 * @param inputs Input tensor of shape (batch_size, input_features)
 * @return Output tensor after all layers, or NULL on error
 */
Tensor *predict(DenseContext *ctx, Tensor *inputs);

/**
 * @brief Computes the accuracy of the network predictions.
 *
 * Computes the accuracy of the network predictions for the given input data
 * and true class labels. The accuracy is computed as the number of correct
 * predictions divided by the total number of predictions.
 *
 * @param ctx Dense layer context containing the layer stack
 * @param pred_class Tensor of predicted class labels (one-hot encoded)
 * @param true_class Tensor of true class labels (one-hot encoded)
 * @return Accuracy of predictions, or NAN on error
 */
float accuracy(DenseContext *ctx, Tensor *pred_class, Tensor *true_class);

#endif // CTORCH_KERAS_H
