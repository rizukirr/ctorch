#ifndef CTORCH_OPS_H
#define CTORCH_OPS_H

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
 * @brief Duplicate input and applies softmax activation function in-place.
 *
 * @param ctx     Memory context for allocation
 * @param input   Source tensor to duplicate and apply activation to
 *
 * @return Output tensor with duplicated input and applied activation, or NULL
 * on error
 */
Tensor *softmax_2dup(TensorContext *ctx, Tensor *input);

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

// Backward pass

/**
 * @brief Computes gradient of cross-entropy loss with respect to logits.
 *
 * For softmax cross-entropy, the gradient simplifies to (output - target) / N
 * where N is the batch size.
 *
 * Math formula: dL/dZ = (softmax(Z) - Y) / N
 *
 * Automatically detects target format based on shape:
 *   - Class indices: target->cols == 1 (shape: batch_size x 1)
 *   - One-hot encoded: target->cols == num_classes (shape: batch_size x
 *     num_classes)
 *
 * @param ctx     Memory context for allocation
 * @param output  Output of softmax activation of shape (batch_size,
 *                num_classes)
 * @param target  Either class indices (batch_size x 1) or one-hot encoded
 *                (batch_size x num_classes)
 *
 * @return Gradient tensor of shape (batch_size, num_classes), or NULL on error
 */
Tensor *cross_entropy_backward(TensorContext *ctx, Tensor *output,
                               Tensor *target);

/**
 * @brief Computes gradient of MSE loss with respect to predictions.
 *
 * The gradient of 0.5 * (prediction - target)^2 is (prediction - target) / N
 * where N is the batch size.
 *
 * Math formula: dL/dy_pred = (prediction - target) / N
 *
 * @param ctx         Memory context for allocation
 * @param prediction  Predicted values tensor of shape (batch_size, features)
 * @param target      Ground truth values tensor of shape (batch_size, features)
 *
 * @return Gradient tensor of shape (batch_size, features), or NULL on error
 */
Tensor *mse_loss_backward(TensorContext *ctx, Tensor *prediction,
                          Tensor *target);

/**
 * @brief Computes gradient of ReLU activation during backpropagation.
 *
 * ReLU derivative is 1 for positive inputs and 0 otherwise. The gradient
 * passes through unchanged where the original input was positive.
 *
 * Math formula: dL/dZ = grad_output * (input > 0 ? 1 : 0)
 *
 * @param ctx          Memory context for allocation
 * @param grad_output  Upstream gradient (dL/da) of shape (batch_size, features)
 * @param input        Original pre-activation inputs to ReLU of shape
 *                     (batch_size, features)
 *
 * @return Gradient tensor of shape (batch_size, features), or NULL on error
 */
Tensor *relu_backward(TensorContext *ctx, Tensor *grad_output, Tensor *input);

/**
 * @brief Computes gradient of sigmoid activation during backpropagation.
 *
 * Sigmoid derivative is sigmoid(x) * (1 - sigmoid(x)). Uses the cached
 * sigmoid output to avoid recomputation.
 *
 * Math formula: dL/dZ = grad_output * output * (1 - output)
 *
 * @param ctx          Memory context for allocation
 * @param grad_output  Upstream gradient (dL/da) of shape (batch_size, features)
 * @param output       Cached output from forward sigmoid pass of shape
 *                     (batch_size, features)
 *
 * @return Gradient tensor of shape (batch_size, features), or NULL on error
 */
Tensor *sigmoid_backward(TensorContext *ctx, Tensor *grad_output,
                         Tensor *output);

/**
 * @brief Computes gradient of tanh activation during backpropagation.
 *
 * Tanh derivative is 1 - tanh(x)^2. Uses the cached tanh output to avoid
 * recomputation.
 *
 * Math formula: dL/dZ = grad_output * (1 - output^2)
 *
 * @param ctx          Memory context for allocation
 * @param grad_output  Upstream gradient (dL/da) of shape (batch_size, features)
 * @param output       Cached output from forward tanh pass of shape
 *                     (batch_size, features)
 *
 * @return Gradient tensor of shape (batch_size, features), or NULL on error
 */
Tensor *tanh_backward(TensorContext *ctx, Tensor *grad_output, Tensor *output);

/**
 * @brief Computes gradient of loss with respect to weights in linear layer.
 *
 * For Y = XW + b, the weight gradient is X^T * grad_output.
 *
 * Math formula: dL/dW = input^T @ grad_output
 *
 * @param ctx          Memory context for allocation
 * @param input        Input tensor X of shape (batch_size, in_features) from
 *                     forward pass
 * @param grad_output  Upstream gradient (dL/dY) of shape (batch_size,
 *                     out_features)
 *
 * @return Weight gradient tensor of shape (in_features, out_features), or NULL
 *         on error
 */
Tensor *weight_gradient(TensorContext *ctx, Tensor *input, Tensor *grad_output);

/**
 * @brief Computes gradient of loss with respect to biases in linear layer.
 *
 * For Y = XW + b, the bias gradient is the sum of upstream gradients
 * along the batch dimension.
 *
 * Math formula: dL/db = sum(grad_output, axis=0)
 *
 * @param ctx          Memory context for allocation
 * @param grad_output  Upstream gradient (dL/dY) of shape (batch_size,
 *                     out_features)
 *
 * @return Bias gradient tensor of shape (1, out_features), or NULL on error
 */
Tensor *bias_gradient(TensorContext *ctx, Tensor *grad_output);

/**
 * @brief Computes gradient of loss with respect to inputs in linear layer.
 *
 * For Y = XW + b, the input gradient is grad_output @ W^T.
 *
 * Math formula: dL/dX = grad_output @ weight^T
 *
 * @param ctx          Memory context for allocation
 * @param grad_output  Upstream gradient (dL/dY) of shape (batch_size,
 *                     out_features)
 * @param weight       Weight tensor W of shape (in_features, out_features)
 *
 * @return Input gradient tensor of shape (batch_size, in_features), or NULL on
 *         error
 */
Tensor *input_gradient(TensorContext *ctx, Tensor *grad_output, Tensor *weight);

#endif // CTORCH_OPS_H
