#ifndef CTORCH_H
#define CTORCH_H

#include "vector.h"

/**
 * @brief Applies affine transformation (linear layer) to input data.
 *
 * Computes the linear transformation: output = inputs * weights + bias
 * This is the fundamental operation for fully connected (dense) neural network layers.
 *
 * Math formula: Y = XW + b
 * where:
 *   - X is the input matrix (N x D_in)
 *   - W is the weight matrix (D_in x D_out)
 *   - b is the bias vector (D_out)
 *   - Y is the output matrix (N x D_out)
 *
 * @param ctx Memory context for allocation
 * @param inputs Input matrix of shape (N, D_in) where N is batch size
 * @param weights Weight matrix of shape (D_in, D_out) or will be transposed if needed
 * @param bias Bias array of length D_out
 * @return Pointer to output vector of shape (N, D_out), or NULL on error
 */
Vector *affine_transform(VectorContext *ctx, Vector *inputs, Vector *weights,
                         double *bias);

/**
 * @brief Applies ReLU (Rectified Linear Unit) activation function in-place.
 *
 * ReLU is defined as: f(x) = max(0, x)
 * It replaces all negative values with zero while keeping positive values unchanged.
 *
 * Math formula: f(x) = { x  if x > 0
 *                      { 0  if x ≤ 0
 *
 * @param inputs Vector to apply activation to (modified in-place)
 * @return 0 on success, -1 on error
 */
int activation_ReLU(Vector *inputs);

/**
 * @brief Applies sigmoid activation function in-place.
 *
 * Sigmoid squashes input values to the range (0, 1), commonly used for
 * binary classification and as a gating mechanism.
 *
 * Math formula: σ(x) = 1 / (1 + e^(-x))
 *
 * @param inputs Vector to apply activation to (modified in-place)
 * @return 0 on success, -1 on error
 */
int activation_sigmoid(Vector *inputs);

/**
 * @brief Applies softmax activation function in-place.
 *
 * Softmax converts a vector of real numbers into a probability distribution.
 * The output values are in range (0, 1) and sum to 1, making it suitable
 * for multi-class classification.
 *
 * Math formula: softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
 *
 * @param inputs Vector to apply activation to (modified in-place)
 * @return 0 on success, -1 on error
 */
int activation_softmax(Vector *inputs);

/**
 * @brief Computes cross-entropy loss between predictions and labels.
 *
 * Cross-entropy measures the difference between two probability distributions.
 * Lower values indicate better predictions. Commonly used with softmax for
 * multi-class classification.
 *
 * Math formula: L = -Σ(y_i * log(p_i))
 * where:
 *   - y_i are the true labels (one-hot encoded)
 *   - p_i are the predicted probabilities
 *
 * @param logits Predicted probability distribution (after softmax)
 * @param labels True labels (one-hot encoded or class indices)
 * @return Cross-entropy loss value
 * @note Currently not implemented
 */
double cross_entropy(Vector *logits, Vector *labels);

#endif // CTORCH_H
