#ifndef CTORCH_OPS_H
#define CTORCH_OPS_H

#include "autograd.h"
#include "tensor.h"

/**
 * @brief Applies linear transformation (dense/fully-connected layer) to input.
 *
 * Computes the linear transformation: output = input @ weight + bias
 * This is the fundamental operation for fully connected (dense) neural network
 * layers. Equivalent to PyTorch's nn.Linear.
 *
 * Math formula: Y = XW + b
 * where:
 *   - X is the input matrix (batch_size x in_features)
 *   - W is the weight matrix (in_features x out_features)
 *   - b is the bias tensor (out_features)
 *   - Y is the output matrix (batch_size x out_features)
 *
 * @param ctx     Memory context for allocation
 * @param input   Input matrix of shape (batch_size, in_features)
 * @param weight  Weight matrix of shape (in_features, out_features) or will be
 *                transposed if needed
 * @param bias    Bias array of length out_features
 *
 * @return Pointer to output tensor of shape (batch_size, out_features), or NULL
 * on error
 */
Tensor *linear(TensorContext *ctx, Tensor *input, Tensor *weight, float *bias);

/**
 * @brief Applies ReLU (Rectified Linear Unit) activation function in-place.
 *
 * ReLU is defined as: f(x) = max(0, x)
 * It replaces all negative values with zero while keeping positive values
 * unchanged.
 *
 * Math formula: f(x) = { x  if x > 0
 *                      { 0  if x ≤ 0
 *
 * @param input  Tensor to apply activation to (modified in-place)
 *
 * @note On error, sets global error context via ctorch_set_error()
 */
void relu(Tensor *input);

/**
 * @brief Applies sigmoid activation function in-place.
 *
 * Sigmoid squashes input values to the range (0, 1), commonly used for
 * binary classification and as a gating mechanism.
 *
 * Math formula: σ(x) = 1 / (1 + e^(-x))
 *
 * @param input  Tensor to apply activation to (modified in-place)
 *
 * @note On error, sets global error context via ctorch_set_error()
 */
void sigmoid(Tensor *input);

/**
 * @brief Applies softmax activation function in-place.
 *
 * Softmax converts a tensor of real numbers into a probability distribution.
 * The output values are in range (0, 1) and sum to 1, making it suitable
 * for multi-class classification.
 *
 * Math formula: softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
 *
 * @param input  Tensor to apply activation to (modified in-place)
 *
 * @note On error, sets global error context via ctorch_set_error()
 */
void softmax(Tensor *input);

/**
 * @brief Duplicate input and applies softmax activation function in-place
 * then return it as a new Tensor.
 *
 * @param ctx     Memory context for allocation
 * @param input   Source tensor to duplicate and apply activation to
 *
 * @return Output tensor with duplicated input and applied activation, or NULL
 * on error
 */
Tensor *softmax_dup(TensorContext *ctx, Tensor *input);

/**
 * @brief Applies hyperbolic tangent (tanh) activation function in-place.
 *
 * Tanh squashes input values to the range (-1, 1), commonly used as an
 * activation function in hidden layers. It is zero-centered, which can
 * help with gradient flow during training.
 *
 * Math formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 *              which equals: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
 *
 * @param input  Tensor to apply activation to (modified in-place)
 *
 * @note On error, sets global error context via ctorch_set_error()
 * @note Named tanh_ to avoid conflict with math.h tanh()
 */
void tanh_(Tensor *input);

/**
 * @brief Computes cross-entropy loss from raw logits (numerically stable).
 *
 * Computes cross-entropy directly from raw logits using the log-sum-exp trick
 * for numerical stability. This is the preferred method as it avoids potential
 * numerical issues from computing softmax separately.
 *
 * Math formula: L = (1/N) Σᵢ [log(Σⱼ exp(zᵢⱼ)) - zᵢ[yᵢ]]
 *              which equals: L = -(1/N) Σᵢ log(softmax(zᵢ)[yᵢ])
 * where:
 *   - zᵢⱼ are the raw logits (before softmax)
 *   - yᵢ is the true class index for sample i
 *   - N is the number of samples
 *
 * @param ctx     Memory context for allocation
 * @param logits  Raw logit values (before softmax) of shape (batch_size,
 *                num_classes)
 * @param target  True class indices as a tensor of length batch_size
 *
 * @return Average cross-entropy loss across all samples, or NaN on error
 *
 * @note Uses log-sum-exp trick: log(Σ exp(z)) = max + log(Σ exp(z - max))
 * @note On error, sets global error context via ctorch_set_error() and returns
 * NaN
 */
Tensor *cross_entropy(TensorContext *ctx, Tensor *logits, Tensor *target);

/**
 * @brief Computes element-wise mean squared error (MSE) loss.
 *
 * Calculates the squared error between predictions and ground truth for
 * regression tasks. Uses the formula with 0.5 factor for cleaner gradients.
 *
 * Math formula: L = 0.5 * (prediction - target)^2
 *
 * @param ctx         Memory context for allocation
 * @param prediction  Predicted values tensor of shape (batch_size, features)
 * @param target      Ground truth values tensor of shape (batch_size, features)
 *
 * @return Tensor containing element-wise squared errors, or NULL on error
 * @note On error, sets global error context via ctorch_set_error()
 */
Tensor *mse_loss(TensorContext *ctx, Tensor *prediction, Tensor *target);

#endif // CTORCH_OPS_H
