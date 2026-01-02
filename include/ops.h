#ifndef CTORCH_OPS_H
#define CTORCH_OPS_H

#include "tensor.h"

/**
 * @brief Computes the average of all elements in a tensor.
 *
 * Calculates the mean value across all elements in the tensor by summing
 * all values and dividing by the total number of elements (rows * cols).
 *
 * Math formula: avg = (Σ all elements) / (rows × cols)
 *
 * @param src Source tensor
 * @return Average value of all elements, or 0.0f on error
 * @note On error, sets global error context via ctorch_set_error()
 */
float tensor_avg(Tensor *src);

/**
 * @brief Computes classification accuracy between predictions and true labels.
 *
 * Calculates the percentage of correct predictions by comparing the predicted
 * class (argmax of y_pred) with the true class labels. Each row in y_pred
 * represents probability distribution over classes, and y_true contains the
 * true class indices.
 *
 * Process:
 *   1. For each sample, find predicted class using argmax(y_pred[i])
 *   2. Compare with true class indices in y_true[i]
 *   3. Count matches and compute percentage
 *
 * Math formula: accuracy = (number of correct predictions / total samples) × 100
 *
 * @param y_pred Predicted probabilities tensor of shape (N, num_classes)
 * @param y_true True class indices tensor of shape (N, K) where K is the
 *               number of labels per sample (typically 1 for single-label
 *               classification)
 * @return Accuracy percentage in range [0.0, 100.0], or 0.0f on error
 * @note On error, sets global error context via ctorch_set_error()
 */
float tensor_accuracy(Tensor *y_pred, Tensor *y_true);

/**
 * @brief Performs matrix multiplication between two tensors.
 *
 * Computes the matrix product of two tensors: output = a * b.
 * The number of columns in `a` must equal the number of rows in `b`.
 *
 * Math formula: C = AB
 * where:
 *   - A is a matrix of shape (m, n)
 *   - B is a matrix of shape (n, p)
 *   - C is the resulting matrix of shape (m, p)
 *
 * @param ctx Memory context for allocation
 * @param a The left-hand side tensor of the multiplication
 * @param b The right-hand side tensor of the multiplication
 * @return Pointer to the output tensor, or NULL on error
 */
Tensor *tensor_mul(TensorContext *ctx, Tensor *a, Tensor *b);

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
 * @note On error, sets global error context via ctorch_set_error()
 */
Tensor *affine_transform(TensorContext *ctx, Tensor *inputs, Tensor *weights,
                         float *bias);

/**
 * @brief Applies ReLU (Rectified Linear Unit) activation function in-place
 * (element-wise).
 *
 * ReLU is defined as: f(x) = max(0, x)
 * It replaces all negative values with zero while keeping positive values
 * unchanged. Each element is processed independently.
 *
 * Math formula: f(x) = { x  if x > 0
 *                      { 0  if x ≤ 0
 *
 * @param inputs Tensor to apply activation to (modified in-place, element-wise)
 * @return 0 on success, negative value on error
 * @note On error, sets global error context via ctorch_set_error()
 */
int relu(Tensor *inputs);

/**
 * @brief Applies sigmoid activation function in-place (element-wise).
 *
 * Sigmoid squashes input values to the range (0, 1), commonly used for
 * binary classification and as a gating mechanism. Each element is processed
 * independently.
 *
 * Math formula: σ(x) = 1 / (1 + e^(-x))
 *
 * @param inputs Tensor to apply activation to (modified in-place, element-wise)
 * @return 0 on success, negative value on error
 * @note On error, sets global error context via ctorch_set_error()
 */
int sigmoid(Tensor *inputs);

/**
 * @brief Applies softmax activation function in-place (row-wise).
 *
 * Softmax converts a tensor of real numbers into a probability distribution.
 * The output values are in range (0, 1) and sum to 1 per row, making it
 * suitable for multi-class classification. Each row is normalized
 * independently.
 *
 * Math formula: softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for all j in the same row
 *
 * @param inputs Tensor to apply activation to (modified in-place, row-wise)
 * @return 0 on success, negative value on error
 * @note On error, sets global error context via ctorch_set_error()
 */
int softmax(Tensor *inputs);

/**
 * @brief Applies hyperbolic tangent (tanh) activation function in-place
 * (element-wise).
 *
 * Tanh squashes input values to the range (-1, 1), commonly used as an
 * activation function in hidden layers. It is zero-centered, which can
 * help with gradient flow during training. Each element is processed
 * independently.
 *
 * Math formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 *              which equals: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
 *
 * @param inputs Tensor to apply activation to (modified in-place, element-wise)
 * @return 0 on success, negative value on error
 * @note On error, sets global error context via ctorch_set_error()
 */
int tanhh(Tensor *inputs);

/**
 * @brief Computes cross-entropy loss from raw logits (numerically stable).
 *
 * Computes cross-entropy directly from raw logits using the log-sum-exp trick
 * for numerical stability. This is the preferred method as it avoids potential
 * numerical issues from computing softmax separately.
 *
 * Math formula: Lᵢⱼ = log(Σₖ exp(zᵢₖ)) - zᵢ[yᵢⱼ]
 *              which equals: Lᵢⱼ = -log(softmax(zᵢ)[yᵢⱼ])
 * where:
 *   - zᵢₖ are the raw logits (before softmax) for sample i, class k
 *   - yᵢⱼ is the j-th true class index for sample i
 *   - Lᵢⱼ is the loss for sample i, label j
 *
 * @param ctx Memory context for allocation
 * @param logits Raw logit values (before softmax) of shape (N, num_classes)
 * @param labels True class indices as a tensor of shape (N, K) where K is the
 *               number of labels per sample (typically 1 for single-label
 *               classification, >1 for multi-label classification)
 * @return Tensor of shape (N*K, 1) containing individual losses, or NULL on
 * error For single-label classification with labels of shape (N, 1), returns
 *         (N, 1) with one loss per sample
 * @note Uses log-sum-exp trick: log(Σ exp(z)) = max + log(Σ exp(z - max))
 * @note On error, sets global error context via ctorch_set_error() and returns
 * NULL
 */
Tensor *cross_entropy(TensorContext *ctx, Tensor *logits, Tensor *labels);

/**
 * @brief Performs element-wise subtraction between two tensors.
 *
 * Computes the element-wise difference of two tensors: result = a - b.
 * The two tensors must have the same shape (same rows and columns).
 *
 * Math formula: Cᵢⱼ = Aᵢⱼ - Bᵢⱼ
 *
 * @param ctx Memory context for allocation
 * @param a The first tensor (minuend)
 * @param b The second tensor (subtrahend)
 * @return Pointer to the output tensor containing element-wise differences, or
 * NULL on error
 * @note On error, sets global error context via ctorch_set_error()
 */
Tensor *tensor_subtract(TensorContext *ctx, Tensor *a, Tensor *b);

// ========== BACKWARD PROPAGATION OPERATIONS ==========

/**
 * @brief Backward pass for ReLU activation function.
 *
 * Computes gradient with respect to input given gradient of output.
 * Formula: dL/dx = dL/dy * (x > 0 ? 1 : 0)
 *
 * @param ctx Memory context for allocation
 * @param grad_output Gradient flowing back from next layer
 * @param pre_activation Pre-activation values (input to ReLU forward pass)
 * @param grad_input Output: gradient with respect to input
 * @return 0 on success, negative error code on failure
 */
int relu_backward(TensorContext *ctx, Tensor *grad_output,
                  Tensor *pre_activation, Tensor *grad_input);

/**
 * @brief Backward pass for sigmoid activation function.
 *
 * Computes gradient with respect to input given gradient of output.
 * Formula: dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))
 *         = dL/dy * output * (1 - output)
 *
 * @param ctx Memory context for allocation
 * @param grad_output Gradient flowing back from next layer
 * @param output Output from sigmoid forward pass (sigmoid(x))
 * @param grad_input Output: gradient with respect to input
 * @return 0 on success, negative error code on failure
 */
int sigmoid_backward(TensorContext *ctx, Tensor *grad_output, Tensor *output,
                     Tensor *grad_input);

/**
 * @brief Backward pass for tanh activation function.
 *
 * Computes gradient with respect to input given gradient of output.
 * Formula: dL/dx = dL/dy * (1 - tanh(x)²)
 *         = dL/dy * (1 - output²)
 *
 * @param ctx Memory context for allocation
 * @param grad_output Gradient flowing back from next layer
 * @param output Output from tanh forward pass (tanh(x))
 * @param grad_input Output: gradient with respect to input
 * @return 0 on success, negative error code on failure
 */
int tanh_backward(TensorContext *ctx, Tensor *grad_output, Tensor *output,
                  Tensor *grad_input);

/**
 * @brief Backward pass for affine transformation (linear layer).
 *
 * Computes gradients with respect to inputs, weights, and bias.
 * Formula:
 *   dL/dX = dL/dY @ W^T
 *   dL/dW = X^T @ dL/dY
 *   dL/db = sum(dL/dY, axis=0)
 *
 * @param ctx Memory context for allocation
 * @param grad_output Gradient flowing back from next layer (dL/dY)
 * @param inputs Input tensor from forward pass (X)
 * @param weights Weight tensor from forward pass (W)
 * @param grad_inputs Output: gradient with respect to inputs (dL/dX)
 * @param grad_weights Output: gradient with respect to weights (dL/dW)
 * @param grad_bias Output: gradient with respect to bias (dL/db)
 * @return grad_inputs tensor on success, NULL on error
 */
Tensor *affine_backward(TensorContext *ctx, Tensor *grad_output, Tensor *inputs,
                        Tensor *weights, Tensor **grad_inputs,
                        Tensor **grad_weights, float **grad_bias);

/**
 * @brief Backward pass for softmax + cross-entropy loss (combined).
 *
 * Computes gradient of combined softmax and cross-entropy loss.
 * This is numerically stable and more efficient than computing separately.
 *
 * Formula: dL/dlogits = (softmax(logits) - one_hot(labels)) / batch_size
 *
 * @param ctx Memory context for allocation
 * @param logits Raw logit values (before softmax) from forward pass
 * @param labels True class indices
 * @return Gradient tensor with respect to logits, or NULL on error
 */
Tensor *cross_entropy_backward(TensorContext *ctx, Tensor *logits,
                               Tensor *labels);

#endif // CTORCH_OPS_H
