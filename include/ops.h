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
 * @param y_true True class indices as a tensor of length N
 * @return Average cross-entropy loss across all samples, or NaN on error
 * @note Uses log-sum-exp trick: log(Σ exp(z)) = max + log(Σ exp(z - max))
 * @note On error, sets global error context via ctorch_set_error() and returns
 * NaN
 */
Tensor *cross_entropy(TensorContext *ctx, Tensor *logits, Tensor *y_true);

/**
 * @brief Computes element-wise squared error loss.
 *
 * Calculates the squared error between predictions and ground truth for
 * regression tasks. Uses the formula with 0.5 factor for cleaner gradients.
 *
 * Math formula: L = 0.5 * (y_pred - y_true)^2
 *
 * @param ctx Memory context for allocation
 * @param logits Predicted values tensor of shape (N, D)
 * @param y_true Ground truth values tensor of shape (N, D)
 * @return Tensor containing element-wise squared errors, or NULL on error
 * @note On error, sets global error context via ctorch_set_error()
 */
Tensor *squared_error(TensorContext *ctx, Tensor *logits, Tensor *y_true);

// Backward pass

/**
 * @brief Computes gradient of cross-entropy loss with respect to logits.
 *
 * For softmax cross-entropy, the gradient simplifies to (softmax_output - y_true) / N
 * where N is the batch size. This assumes y_true is one-hot encoded.
 *
 * Math formula: dL/dZ = (softmax(Z) - Y) / N
 *
 * @param ctx Memory context for allocation
 * @param softmax_output Output of softmax activation of shape (N, num_classes)
 * @param y_true One-hot encoded ground truth of shape (N, num_classes)
 * @return Gradient tensor of shape (N, num_classes), or NULL on error
 */
Tensor *cross_entropy_backward(TensorContext *ctx, Tensor *softmax_output, Tensor *y_true);

/**
 * @brief Computes gradient of squared error loss with respect to predictions.
 *
 * The gradient of 0.5 * (y_pred - y_true)^2 is (y_pred - y_true) / N
 * where N is the batch size.
 *
 * Math formula: dL/dy_pred = (y_pred - y_true) / N
 *
 * @param ctx Memory context for allocation
 * @param y_pred Predicted values tensor of shape (N, D)
 * @param y_true Ground truth values tensor of shape (N, D)
 * @return Gradient tensor of shape (N, D), or NULL on error
 */
Tensor *squared_error_backward(TensorContext *ctx, Tensor *y_pred, Tensor *y_true);

/**
 * @brief Computes gradient of ReLU activation during backpropagation.
 *
 * ReLU derivative is 1 for positive inputs and 0 otherwise. The gradient
 * passes through unchanged where the original input was positive.
 *
 * Math formula: dL/dZ = dL/da * (Z > 0 ? 1 : 0)
 *
 * @param ctx Memory context for allocation
 * @param loss_grad Upstream gradient (dL/da) of shape (N, D)
 * @param logits Original pre-activation inputs to ReLU of shape (N, D)
 * @return Gradient tensor of shape (N, D), or NULL on error
 */
Tensor *relu_backward(TensorContext *ctx, Tensor *loss_grad, Tensor *logits);

/**
 * @brief Computes gradient of sigmoid activation during backpropagation.
 *
 * Sigmoid derivative is sigmoid(x) * (1 - sigmoid(x)). Uses the cached
 * sigmoid output to avoid recomputation.
 *
 * Math formula: dL/dZ = dL/da * sigmoid(Z) * (1 - sigmoid(Z))
 *
 * @param ctx Memory context for allocation
 * @param upstream_grad Upstream gradient (dL/da) of shape (N, D)
 * @param sigmoid_output Cached output from forward sigmoid pass of shape (N, D)
 * @return Gradient tensor of shape (N, D), or NULL on error
 */
Tensor *sigmoid_backward(TensorContext *ctx, Tensor *upstream_grad, Tensor *sigmoid_output);

/**
 * @brief Computes gradient of tanh activation during backpropagation.
 *
 * Tanh derivative is 1 - tanh(x)^2. Uses the cached tanh output to avoid
 * recomputation.
 *
 * Math formula: dL/dZ = dL/da * (1 - tanh(Z)^2)
 *
 * @param ctx Memory context for allocation
 * @param upstream_grad Upstream gradient (dL/da) of shape (N, D)
 * @param tanh_output Cached output from forward tanh pass of shape (N, D)
 * @return Gradient tensor of shape (N, D), or NULL on error
 */
Tensor *tanh_backward(TensorContext *ctx, Tensor *upstream_grad, Tensor *tanh_output);

/**
 * @brief Computes gradient of loss with respect to weights in affine layer.
 *
 * For Y = XW + b, the weight gradient is X^T * upstream_grad.
 *
 * Math formula: dL/dW = X^T * dL/dY
 *
 * @param ctx Memory context for allocation
 * @param inputs Input tensor X of shape (N, D_in) from forward pass
 * @param upstream_grad Upstream gradient (dL/dY) of shape (N, D_out)
 * @return Weight gradient tensor of shape (D_in, D_out), or NULL on error
 */
Tensor *weight_gradient(TensorContext *ctx, Tensor *inputs, Tensor *upstream_grad);

/**
 * @brief Computes gradient of loss with respect to biases in affine layer.
 *
 * For Y = XW + b, the bias gradient is the sum of upstream gradients
 * along the batch dimension.
 *
 * Math formula: dL/db = sum(dL/dY, axis=0)
 *
 * @param ctx Memory context for allocation
 * @param upstream_grad Upstream gradient (dL/dY) of shape (N, D_out)
 * @return Bias gradient tensor of shape (1, D_out), or NULL on error
 */
Tensor *bias_gradient(TensorContext *ctx, Tensor *upstream_grad);

/**
 * @brief Computes gradient of loss with respect to inputs in affine layer.
 *
 * For Y = XW + b, the input gradient is upstream_grad * W^T.
 *
 * Math formula: dL/dX = dL/dY * W^T
 *
 * @param ctx Memory context for allocation
 * @param upstream_grad Upstream gradient (dL/dY) of shape (N, D_out)
 * @param weights Weight tensor W of shape (D_in, D_out)
 * @return Input gradient tensor of shape (N, D_in), or NULL on error
 */
Tensor *input_gradient(TensorContext *ctx, Tensor *upstream_grad, Tensor *weights);


#endif // CTORCH_OPS_H
