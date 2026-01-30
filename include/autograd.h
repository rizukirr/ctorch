#ifndef CTORCH_AUTOGRAD_H
#define CTORCH_AUTOGRAD_H

#include "tensor.h"

/**
 * @brief Computes the gradient of cross-entropy loss with respect to logits.
 *
 * Combines the softmax and cross-entropy backward passes into a single
 * efficient computation. The gradient is: softmax(logits) - one_hot(target).
 *
 * Supports two target formats:
 *   - Class indices: target shape (batch_size, 1) with integer class labels
 *   - One-hot encoded: target shape (batch_size, num_classes)
 *
 * The gradient is averaged over the batch (divided by batch_size).
 *
 * @param ctx     Memory context for tensor allocation
 * @param output  Network output after softmax, shape (batch_size, num_classes)
 * @param target  True labels as class indices or one-hot encoded
 *
 * @return Gradient tensor of shape (batch_size, num_classes), or NULL on error
 */
Tensor *cross_entropy_backward(TensorContext *ctx, Tensor *output,
                               Tensor *target);

/**
 * @brief Computes the gradient of MSE loss with respect to predictions.
 *
 * The gradient of 0.5 * (pred - target)^2 is (pred - target).
 * The result is averaged over the batch.
 *
 * Supports two target formats:
 *   - Single value: target shape (batch_size, 1), broadcasted to all columns
 *   - Multi-value: target shape matches prediction shape
 *
 * @param ctx         Memory context for tensor allocation
 * @param prediction  Network output, shape (batch_size, out_features)
 * @param target      True values
 *
 * @return Gradient tensor matching prediction shape, or NULL on error
 */
Tensor *mse_loss_backward(TensorContext *ctx, Tensor *prediction,
                          Tensor *target);

/**
 * @brief Computes the gradient of ReLU activation.
 *
 * ReLU derivative: 1 if input > 0, else 0.
 * The gradient is: upstream * (input > 0 ? 1 : 0)
 *
 * @param ctx          Memory context for tensor allocation
 * @param grad_output  Gradient from upstream layer
 * @param input        Original input to ReLU (pre-activation)
 *
 * @return Gradient tensor matching input shape, or NULL on error
 */
Tensor *relu_backward(TensorContext *ctx, Tensor *grad_output, Tensor *input);

/**
 * @brief Computes the gradient of sigmoid activation.
 *
 * Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
 * The gradient is: upstream * output * (1 - output)
 *
 * @param ctx          Memory context for tensor allocation
 * @param grad_output  Gradient from upstream layer
 * @param output       Sigmoid output (post-activation, not input)
 *
 * @return Gradient tensor matching output shape, or NULL on error
 */
Tensor *sigmoid_backward(TensorContext *ctx, Tensor *grad_output,
                         Tensor *output);

/**
 * @brief Computes the gradient of tanh activation.
 *
 * Tanh derivative: 1 - tanh(x)^2
 * The gradient is: upstream * (1 - output^2)
 *
 * @param ctx          Memory context for tensor allocation
 * @param grad_output  Gradient from upstream layer
 * @param output       Tanh output (post-activation, not input)
 *
 * @return Gradient tensor matching output shape, or NULL on error
 */
Tensor *tanh_backward(TensorContext *ctx, Tensor *grad_output, Tensor *output);

/**
 * @brief Computes the gradient of loss with respect to layer weights.
 *
 * For a linear layer Y = X @ W + b, the weight gradient is:
 *   dL/dW = X^T @ dL/dY
 *
 * @param ctx          Memory context for tensor allocation
 * @param input        Layer input X, shape (batch_size, in_features)
 * @param grad_output  Gradient dL/dY, shape (batch_size, out_features)
 *
 * @return Weight gradient, shape (in_features, out_features), or NULL on error
 */
Tensor *weight_gradient(TensorContext *ctx, Tensor *input, Tensor *grad_output);

/**
 * @brief Computes the gradient of loss with respect to layer biases.
 *
 * For a linear layer Y = X @ W + b, the bias gradient is:
 *   dL/db = sum(dL/dY, axis=0)
 *
 * @param ctx          Memory context for tensor allocation
 * @param grad_output  Gradient dL/dY, shape (batch_size, out_features)
 *
 * @return Bias gradient, shape (1, out_features), or NULL on error
 */
Tensor *bias_gradient(TensorContext *ctx, Tensor *grad_output);

/**
 * @brief Computes the gradient of loss with respect to layer input.
 *
 * For a linear layer Y = X @ W + b, the input gradient is:
 *   dL/dX = dL/dY @ W^T
 *
 * @param ctx          Memory context for tensor allocation
 * @param grad_output  Gradient dL/dY, shape (batch_size, out_features)
 * @param weight       Layer weights W, shape (in_features, out_features)
 *
 * @return Input gradient, shape (batch_size, in_features), or NULL on error
 */
Tensor *input_gradient(TensorContext *ctx, Tensor *grad_output, Tensor *weight);

#endif // CTORCH_AUTOGRAD_H
