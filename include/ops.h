#ifndef CTORCH_OPS_H
#define CTORCH_OPS_H

#include "tensor.h"

/**
 * @brief Applies affine transformation (linear layer) to input data.
 *
 * Computes the linear transformation: output = inputs * weights + bias
 * This is the fundamental operation for fully connected (dense) neural network
 * layers.
 *
 * Math formula: Y = XW + b
 * where:
 *   - X is the input matrix (N x D_in)
 *   - W is the weight matrix (D_in x D_out)
 *   - b is the bias tensor (D_out)
 *   - Y is the output matrix (N x D_out)
 *
 * @param ctx Memory context for allocation
 * @param inputs Input matrix of shape (N, D_in) where N is batch size
 * @param weights Weight matrix of shape (D_in, D_out) or will be transposed if
 * needed
 * @param bias Bias array of length D_out
 * @return Pointer to output tensor of shape (N, D_out), or NULL on error
 */
Tensor *affine_transform(TensorContext *ctx, Tensor *inputs, Tensor *weights,
                         float *bias);

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
 * @param inputs Tensor to apply activation to (modified in-place)
 * @note On error, sets global error context via ctorch_set_error()
 */
void relu(Tensor *inputs);

/**
 * @brief Applies sigmoid activation function in-place.
 *
 * Sigmoid squashes input values to the range (0, 1), commonly used for
 * binary classification and as a gating mechanism.
 *
 * Math formula: σ(x) = 1 / (1 + e^(-x))
 *
 * @param inputs Tensor to apply activation to (modified in-place)
 * @note On error, sets global error context via ctorch_set_error()
 */
void sigmoid(Tensor *inputs);

/**
 * @brief Applies softmax activation function in-place.
 *
 * Softmax converts a tensor of real numbers into a probability distribution.
 * The output values are in range (0, 1) and sum to 1, making it suitable
 * for multi-class classification.
 *
 * Math formula: softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
 *
 * @param inputs Tensor to apply activation to (modified in-place)
 * @note On error, sets global error context via ctorch_set_error()
 */
void softmax(Tensor *inputs);

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
 * @param inputs Tensor to apply activation to (modified in-place)
 * @note On error, sets global error context via ctorch_set_error()
 */
void tanhh(Tensor *inputs);

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
 * @param logits Raw logit values (before softmax) of shape (N, num_classes)
 * @param labels True class indices as a tensor of length N
 * @return Average cross-entropy loss across all samples, or NaN on error
 * @note Uses log-sum-exp trick: log(Σ exp(z)) = max + log(Σ exp(z - max))
 * @note On error, sets global error context via ctorch_set_error() and returns NaN
 */
float cross_entropy(Tensor *logits, Tensor *labels);

#endif // CTORCH_OPS_H
