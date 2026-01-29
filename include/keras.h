#ifndef CTORCH_KERAS_H
#define CTORCH_KERAS_H

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

/**
 * @brief Optimizer algorithm types for weight updates.
 *
 * Defines the strategy used to update weights during backpropagation:
 *
 *   - SGD: Basic gradient descent with fixed learning rate
 *   - Momentum: Accumulates velocity from past gradients (β=0.9)
 *   - RMSprop: Adapts learning rate per-parameter using squared gradient
 *     average (β=0.999)
 *   - Adam: Combines momentum and RMSprop with bias correction (β₁=0.9,
 *     β₂=0.999)
 */
typedef enum { Adam, SGD, Momentum, RMSprop } OptimizerType;

typedef struct DenseContext DenseContext;
typedef struct OptimizerState OptimizerState;

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
 * @brief Performs Stochastic Gradient Descent (SGD) on the given layer.
 *
 * Updates weights and biases using the simplest optimization rule:
 *
 *   weight = weight - lr * gradient
 *
 * SGD has no memory of past gradients. Each update depends only on the
 * current gradient, which can cause oscillation in steep valleys.
 *
 * @param lr     Learning rate (step size for gradient descent)
 * @param layer  Layer to update weights and biases for
 * @param dw     Gradient of loss with respect to layer weights (dL/dW)
 * @param db     Gradient of loss with respect to layer biases (dL/db)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int sgd(float lr, Dense *layer, Tensor *dw, Tensor *db);

/**
 * @brief Initializes a new OptimizerState for a given layer.
 *
 * Creates optimizer state tensors for storing moment estimates used by
 * Momentum, RMSprop, and Adam optimizers. The state contains:
 *   - m_weights, m_biases: First moment (mean of gradients) for Momentum/Adam
 *   - v_weights, v_biases: Second moment (mean of squared gradients) for
 *                          RMSprop/Adam
 *   - t: Timestep counter for Adam bias correction
 *
 * All moment tensors are initialized to zeros. Shape matches layer parameters:
 *   - m_weights, v_weights: (in_features x out_features)
 *   - m_biases, v_biases: (1 x out_features)
 *
 * @param ctx           Memory context for tensor allocation
 * @param in_features   Number of input features (rows in weight matrix)
 * @param out_features  Number of output features (cols in weight matrix)
 *
 * @return Pointer to new OptimizerState, or NULL on allocation failure
 */
OptimizerState *optimizer_state_init(DenseContext *ctx, size_t in_features,
                                     size_t out_features);

/**
 * @brief Performs Momentum optimization on the given layer.
 *
 * Accumulates velocity from past gradients to smooth updates and accelerate
 * convergence. Helps escape local minima and reduces oscillation in valleys.
 *
 * Formula:
 *   v = β*v + (1-β)*g        (update velocity)
 *   weight = weight - lr*v   (update parameters)
 *
 * Where:
 *   β = 0.9 (momentum coefficient)
 *   g = gradient (dw or db)
 *   v = velocity (stored in optimizer_state, persists across iterations)
 *
 * @param optimizer_state  Optimizer state containing velocity tensors
 * (v_weights, v_biases)
 * @param lr               Learning rate
 * @param layer            Layer to update weights and biases for
 * @param dw               Gradient of loss with respect to layer weights
 * (dL/dW)
 * @param db               Gradient of loss with respect to layer biases (dL/db)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int momentum(OptimizerState *optimizer_state, float lr, Dense *layer,
             Tensor *dw, Tensor *db);

/**
 * @brief Performs RMSprop (Root Mean Square Propagation) on the given layer.
 *
 * Adapts learning rate per-parameter by dividing by the running average of
 * gradient magnitudes. Parameters with large gradients get smaller updates,
 * parameters with small gradients get larger updates.
 *
 * Formula:
 *   v = β*v + (1-β)*g²              (update squared gradient average)
 *   weight = weight - lr*g/(√v+ε)   (update parameters)
 *
 * Where:
 *   β = 0.999 (decay rate)
 *   ε = 1e-8 (prevents division by zero)
 *   g = gradient (dw or db)
 *   v = running average of squared gradients (stored in optimizer_state)
 *
 * @param optimizer_state  Optimizer state containing squared gradient averages
 *                         (v_weights, v_biases)
 * @param lr               Learning rate
 * @param layer            Layer to update weights and biases for
 * @param dw               Gradient of loss with respect to layer weights
 *                         (dL/dW)
 * @param db               Gradient of loss with respect to layer biases (dL/db)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int rmsprop(OptimizerState *optimizer_state, float lr, Dense *layer, Tensor *dw,
            Tensor *db);

/**
 * @brief Performs Adam (Adaptive Moment Estimation) on the given layer.
 *
 * Combines Momentum and RMSprop with bias correction. Tracks both the mean
 * (first moment) and variance (second moment) of gradients, providing adaptive
 * per-parameter learning rates with smoothed updates.
 *
 * Formula:
 *   m = β₁*m + (1-β₁)*g           (first moment estimate)
 *   v = β₂*v + (1-β₂)*g²          (second moment estimate)
 *   m̂ = m / (1-β₁ᵗ)               (bias-corrected first moment)
 *   v̂ = v / (1-β₂ᵗ)               (bias-corrected second moment)
 *   weight = weight - lr*m̂/(√v̂+ε) (update parameters)
 *
 * Where:
 *   β₁ = 0.9 (first moment decay)
 *   β₂ = 0.999 (second moment decay)
 *   ε = 1e-8 (prevents division by zero)
 *   t = timestep (increments each call, stored in optimizer_state)
 *
 * Bias correction compensates for zero-initialization of moments, especially
 * important in early training iterations.
 *
 * @param optimizer_state  Optimizer state containing moment estimates
 *                         (m_weights, v_weights, m_biases, v_biases, t)
 * @param lr               Learning rate
 * @param layer            Layer to update weights and biases for
 * @param dw               Gradient of loss with respect to layer weights
 *                         (dL/dW)
 * @param db               Gradient of loss with respect to layer biases (dL/db)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int adam(OptimizerState *optimizer_state, float lr, Dense *layer, Tensor *dw,
         Tensor *db);

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
