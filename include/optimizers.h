#ifndef CTORCH_OPTIMIZERS_H
#define CTORCH_OPTIMIZERS_H

#include "tensor.h"

/* Optimizer hyperparameters */
#define MOMENTUM_BETA 0.9f  // Momentum decay rate for velocity
#define RMSPROP_BETA 0.999f // RMSprop decay rate for squared gradient average
#define EPSILON 1e-8f       // Small constant to prevent division by zero
#define ADAM_BETA1 0.9f     // Adam first moment decay rate
#define ADAM_BETA2 0.999f   // Adam second moment decay rate

/**
 * @brief Optimizer algorithm types for weight updates.
 *
 * Defines the strategy used to update weights during backpropagation:
 *
 *   - SGD: Basic gradient descent with fixed learning rate
 *   - Momentum: Accumulates velocity from past gradients (beta=0.9)
 *   - RMSprop: Adapts learning rate per-parameter using squared gradient
 *     average (beta=0.999)
 *   - Adam: Combines momentum and RMSprop with bias correction (beta1=0.9,
 *     beta2=0.999)
 */
typedef enum { Adam, SGD, Momentum, RMSprop } OptimizerType;

/**
 * @brief Optimizer state for adaptive gradient methods.
 *
 * Stores moment estimates and correction terms needed by Momentum, RMSprop,
 * and Adam optimizers. Each Dense layer has its own OptimizerState to track
 * per-parameter statistics across training iterations.
 *
 * Memory layout:
 *   - v_weights, v_biases: Second moment (mean of squared gradients) for
 *     RMSprop/Adam, or velocity for Momentum
 *   - m_weights, m_biases: First moment (mean of gradients) for Adam only
 *   - t: Timestep counter incremented each optimizer step
 *   - beta1_correction: Accumulated beta1^t for Adam bias correction (starts at
 *     1.0, multiplied by beta1 each step)
 *   - beta2_correction: Accumulated beta2^t for Adam bias correction (starts at
 *     1.0, multiplied by beta2 each step)
 *
 * Shape invariants:
 *   - v_weights, m_weights: (in_features x out_features) matching layer weight
 *   - v_biases, m_biases: (1 x out_features) matching layer bias
 *
 * @field v_weights         Second moment estimate for weights (or velocity for
 *                          Momentum)
 * @field v_biases          Second moment estimate for biases (or velocity for
 *                          Momentum)
 * @field m_weights         First moment estimate for weights (Adam only)
 * @field m_biases          First moment estimate for biases (Adam only)
 * @field t                 Timestep counter (incremented each update)
 * @field beta1_correction  Accumulated beta1^t for bias correction (Adam only)
 * @field beta2_correction  Accumulated beta2^t for bias correction (Adam only)
 */
typedef struct {
  Tensor *v_weights;
  Tensor *v_biases;
  Tensor *m_weights;
  Tensor *m_biases;
  int t;
  float beta1_correction;
  float beta2_correction;
} OptimizerState;

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
OptimizerState *optimizer_state_init(TensorContext *ctx, size_t in_features,
                                     size_t out_features);

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
 * @param lr      Learning rate (step size for gradient descent)
 * @param weight  Weight tensor to update
 * @param bias    Bias tensor to update
 * @param dw      Gradient of loss with respect to weights (dL/dW)
 * @param db      Gradient of loss with respect to biases (dL/db)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int sgd(float lr, Tensor *weight, Tensor *bias, Tensor *dw, Tensor *db);

/**
 * @brief Performs Momentum optimization on the given layer.
 *
 * Accumulates velocity from past gradients to smooth updates and accelerate
 * convergence. Helps escape local minima and reduces oscillation in valleys.
 *
 * Formula:
 *   v = beta*v + (1-beta)*g        (update velocity)
 *   weight = weight - lr*v         (update parameters)
 *
 * Where:
 *   beta = 0.9 (momentum coefficient)
 *   g = gradient (dw or db)
 *   v = velocity (stored in optimizer_state, persists across iterations)
 *
 * @param state   Optimizer state containing velocity tensors (v_weights,
 *                v_biases)
 * @param lr      Learning rate
 * @param weight  Weight tensor to update
 * @param bias    Bias tensor to update
 * @param dw      Gradient of loss with respect to weights (dL/dW)
 * @param db      Gradient of loss with respect to biases (dL/db)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int momentum(OptimizerState *state, float lr, Tensor *weight, Tensor *bias,
             Tensor *dw, Tensor *db);

/**
 * @brief Performs RMSprop (Root Mean Square Propagation) on the given layer.
 *
 * Adapts learning rate per-parameter by dividing by the running average of
 * gradient magnitudes. Parameters with large gradients get smaller updates,
 * parameters with small gradients get larger updates.
 *
 * Formula:
 *   v = beta*v + (1-beta)*g^2            (update squared gradient average)
 *   weight = weight - lr*g/(sqrt(v)+eps) (update parameters)
 *
 * Where:
 *   beta = 0.999 (decay rate)
 *   eps = 1e-8 (prevents division by zero)
 *   g = gradient (dw or db)
 *   v = running average of squared gradients (stored in optimizer_state)
 *
 * @param state   Optimizer state containing squared gradient averages
 *                (v_weights, v_biases)
 * @param lr      Learning rate
 * @param weight  Weight tensor to update
 * @param bias    Bias tensor to update
 * @param dw      Gradient of loss with respect to weights (dL/dW)
 * @param db      Gradient of loss with respect to biases (dL/db)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int rmsprop(OptimizerState *state, float lr, Tensor *weight, Tensor *bias,
            Tensor *dw, Tensor *db);

/**
 * @brief Performs Adam (Adaptive Moment Estimation) on the given layer.
 *
 * Combines Momentum and RMSprop with bias correction. Tracks both the mean
 * (first moment) and variance (second moment) of gradients, providing adaptive
 * per-parameter learning rates with smoothed updates.
 *
 * Formula:
 *   m = beta1*m + (1-beta1)*g           (first moment estimate)
 *   v = beta2*v + (1-beta2)*g^2         (second moment estimate)
 *   m_hat = m / (1-beta1^t)             (bias-corrected first moment)
 *   v_hat = v / (1-beta2^t)             (bias-corrected second moment)
 *   weight = weight - lr*m_hat/(sqrt(v_hat)+eps) (update parameters)
 *
 * Where:
 *   beta1 = 0.9 (first moment decay)
 *   beta2 = 0.999 (second moment decay)
 *   eps = 1e-8 (prevents division by zero)
 *   t = timestep (increments each call, stored in optimizer_state)
 *
 * Bias correction compensates for zero-initialization of moments, especially
 * important in early training iterations.
 *
 * @param state   Optimizer state containing moment estimates
 *                (m_weights, v_weights, m_biases, v_biases, t)
 * @param lr      Learning rate
 * @param weight  Weight tensor to update
 * @param bias    Bias tensor to update
 * @param dw      Gradient of loss with respect to weights (dL/dW)
 * @param db      Gradient of loss with respect to biases (dL/db)
 *
 * @return 0 on success, negative CTorchError code on failure
 */
int adam(OptimizerState *state, float lr, Tensor *weight, Tensor *bias,
         Tensor *dw, Tensor *db);

#endif // CTORCH_OPTIMIZERS_H
