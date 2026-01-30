#ifndef CTORCH_KERAS_H
#define CTORCH_KERAS_H

#include "optimizers.h"
#include "tensor.h"

/**
 * @brief Activation function types for neural network layers.
 *
 * Defines the nonlinear transformations applied after linear layer operations.
 * Each activation has different properties suited for different use cases:
 *
 *   - Linear: Identity function (no activation). Used for regression outputs.
 *   - ReLU: f(x) = max(0, x). Most common for hidden layers, prevents vanishing
 *     gradients.
 *   - Sigmoid: f(x) = 1/(1+e^-x). Squashes to [0,1], used for binary
 *     classification.
 *   - Softmax: f(x_i) = e^x_i / Σe^x_j. Normalizes to probability distribution,
 *     used for multi-class classification.
 *   - Tanh: f(x) = tanh(x). Squashes to [-1,1], zero-centered alternative to
 *     sigmoid.
 */
typedef enum { Linear, ReLU, Sigmoid, Softmax, Tanh } Activation;

/**
 * @brief Loss function types for training.
 *
 * Defines how prediction error is measured during training:
 *
 *   - CrossEntropy: -Σ(y * log(ŷ)). Standard for classification tasks, measures
 *     divergence between predicted and true probability distributions.
 */
typedef enum { CrossEntropy } Loss;

typedef struct DenseContext DenseContext;

/**
 * @brief Dense (fully connected) layer structure.
 *
 * Represents a single dense layer in a neural network. Performs linear
 * transformation followed by an activation function:
 *
 *   output = activation(input @ weight + bias)
 *
 * Where:
 *   - input: (batch_size × in_features)
 *   - weight: (in_features × out_features)
 *   - bias: (1 × out_features) - broadcasted across batch
 *   - output: (batch_size × out_features)
 *
 * Weight initialization depends on activation type:
 *   - ReLU: He initialization (accounts for ReLU's asymmetry)
 *   - Sigmoid/Tanh/Linear/Softmax: Xavier initialization (symmetric
 *     activations)
 *
 * @field weight      Weight matrix of shape (in_features × out_features)
 * @field bias        Bias vector of shape (1 × out_features)
 * @field activation  Activation function applied after linear transformation
 */
typedef struct {
  Tensor *weight;
  Tensor *bias;
  Activation activation;
} Dense;

/**
 * @brief Initializes a new Dense layer context.
 *
 * Creates a context for managing Dense layer allocations using arena allocator.
 * The input size is tracked so that subsequent layers can be created with
 * compatible dimensions.
 *
 * @param in_features   Number of input features for the first layer in this
 *                      context
 *
 * @return Pointer to new DenseContext, or NULL on allocation failure
 */
DenseContext *dense_init(size_t in_features);

/**
 * @brief Creates a new Dense (fully connected) layer.
 *
 * Allocates and initializes a dense layer with random weights and biases.
 * The input size is determined from the context (set by dense_init() or
 * updated by previous layer creation). The layer is added to the context's
 * internal layer stack.
 *
 * @param ctx           Dense layer context containing the layer stack
 * @param out_features  Number of output features (neurons) in this layer
 * @param activation    Activation function to apply after this layer
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int dense_create(DenseContext *ctx, size_t out_features, Activation activation);

/**
 * @brief Performs forward pass through all Dense layers in the context.
 *
 * Runs the input through each layer in sequence, applying linear transformation
 * (output = input @ weight + bias) followed by each layer's activation
 * function. The final output is cached in ctx->output.
 *
 * @param ctx    Dense layer context containing the layer stack
 * @param input  Input tensor of shape (batch_size, in_features)
 *
 * @return Output tensor after all layers, or NULL on error
 */
Tensor *dense_forward(DenseContext *ctx, Tensor *input);

/**
 * @brief Frees a Dense layer context and all associated layers.
 *
 * Releases all memory allocated for the Dense context, including all layers
 * created with this context.
 *
 * @param ctx  DenseContext to free
 */
void dense_free(DenseContext *ctx);

/**
 * @brief Performs backward pass (backpropagation) through all Dense layers.
 *
 * Computes gradients and updates weights/biases for all layers using the
 * cached intermediate values from the forward pass. Must be called after
 * dense_forward().
 *
 * @param ctx           Dense layer context containing the layer stack and
 *                      output tensor
 * @param lr            Learning rate (step size for gradient descent weight
 *                      updates)
 * @param grad_output   Gradient of loss with respect to network output (dL/dY)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int dense_backward(DenseContext *ctx, Tensor *grad_output, float lr,
                   OptimizerType optimizer);

/**
 * @brief Computes predictions for the given input data.
 *
 * Computes the output of the network for the given input data. The output is
 * a tensor containing the predicted class index for each sample.
 *
 * @param ctx    Dense layer context containing the layer stack
 * @param input  Input tensor of shape (batch_size, in_features)
 *
 * @return Output tensor with predicted class indices, or NULL on error
 */
Tensor *predict(DenseContext *ctx, Tensor *input);

/**
 * @brief Computes the accuracy of the network predictions.
 *
 * Computes the accuracy of the network predictions for the given input data
 * and true class labels. The accuracy is computed as the number of correct
 * predictions divided by the total number of predictions.
 *
 * @param ctx           Dense layer context containing the layer stack
 * @param predictions   Tensor of predicted class labels
 * @param targets       Tensor of true class labels (can be one-hot encoded or
 *                      indices)
 *
 * @return Accuracy of predictions, or NAN on error
 */
float accuracy(DenseContext *ctx, Tensor *predictions, Tensor *targets);

#endif // CTORCH_KERAS_H
