#include "keras.h"
#include "arena.h"
#include "errors.h"
#include "ops.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define dense_ARENA_SIZE 1024 * 16

/* Optimizer hyperparameters */
#define MOMENTUM_BETA 0.9f  // Momentum decay rate for velocity
#define RMSPROP_BETA 0.999f // RMSprop decay rate for squared gradient average
#define EPSILON 1e-8f       // Small constant to prevent division by zero
#define ADAM_BETA1 0.9f     // Adam first moment decay rate
#define ADAM_BETA2 0.999f   // Adam second moment decay rate

/**
 * @brief Dynamic array of Dense layer pointers.
 *
 * Used to store the sequential stack of Dense layers in a neural network.
 * Supports dynamic resizing as layers are added during model construction.
 *
 * @field capacity  Maximum number of layers the array can hold before resizing
 * @field size      Current number of layers stored in the array
 * @field items     Array of pointers to Dense layer structures
 */
typedef struct {
  size_t capacity;
  size_t size;
  Dense **items;
} DenseArr;

/**
 * @brief Dynamic array of Tensor pointers.
 *
 * Used to cache intermediate tensors during forward pass (layer inputs and
 * pre-activation values). These cached values are needed during backpropagation
 * to compute gradients.
 *
 * @field capacity  Maximum number of tensors the array can hold before resizing
 * @field size      Current number of tensors stored in the array
 * @field items     Array of pointers to Tensor structures
 */
typedef struct {
  size_t capacity;
  size_t size;
  Tensor **items;
} TensorArr;

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
 *   - beta1_correction: Accumulated β₁^t for Adam bias correction (starts at
 *     1.0, multiplied by β₁ each step)
 *   - beta2_correction: Accumulated β₂^t for Adam bias correction (starts at
 *     1.0, multiplied by β₂ each step)
 *
 * Shape invariants:
 *   - v_weights, m_weights: (in_features × out_features) matching layer weight
 *   - v_biases, m_biases: (1 × out_features) matching layer bias
 *
 * @field v_weights         Second moment estimate for weights (or velocity for
 *                          Momentum)
 * @field v_biases          Second moment estimate for biases (or velocity for
 *                          Momentum)
 * @field m_weights         First moment estimate for weights (Adam only)
 * @field m_biases          First moment estimate for biases (Adam only)
 * @field t                 Timestep counter (incremented each update)
 * @field beta1_correction  Accumulated β₁^t for bias correction (Adam only)
 * @field beta2_correction  Accumulated β₂^t for bias correction (Adam only)
 */
struct OptimizerState {
  Tensor *v_weights;
  Tensor *v_biases;
  Tensor *m_weights;
  Tensor *m_biases;
  int t;
  float beta1_correction;
  float beta2_correction;
};

/**
 * @brief Dynamic array of OptimizerState pointers.
 *
 * Stores optimizer state for each Dense layer in the network. The array is
 * parallel to the DenseArr - optimizer_state[i] corresponds to
 * hidden_layers[i].
 *
 * @field capacity  Maximum number of optimizer states before resizing
 * @field size      Current number of optimizer states (matches layer count)
 * @field items     Array of pointers to OptimizerState structures
 */
typedef struct {
  size_t capacity;
  size_t size;
  OptimizerState **items;
} OptimizerStateArr;

/**
 * @brief Sequential neural network context (Keras-style model).
 *
 * Manages a stack of Dense layers and their associated state for
 * forward/backward passes. Implements a sequential feedforward architecture
 * where each layer's output becomes the next layer's input.
 *
 * Memory management:
 *   - All layers, optimizer states, and dynamic arrays are allocated via arena
 *   - Intermediate tensors (layer_inputs, pre_activations, output) are managed
 *     by tensor_ctx and freed together
 *   - Call dense_free() to release all resources at once
 *
 * Usage pattern:
 *   1. dense_init(in_features) - create context
 *   2. dense_create(...) - add layers sequentially
 *   3. dense_forward(input) - compute predictions
 *   4. dense_backward(grad_output, lr, optimizer) - update weights
 *   5. dense_free() - cleanup
 *
 * @field tensor_ctx       Memory context for all tensor allocations
 * @field input_size       Expected number of input features for next layer
 * (updated after each dense_create call)
 * @field hidden_layers    Sequential stack of Dense layers in the network
 * @field layer_inputs     Cached inputs to each layer during forward pass
 * (needed for weight gradients in backprop)
 * @field pre_activations  Cached pre-activation outputs (after linear
 * transform, before activation function) for each layer
 * @field optimizer_state  Optimizer state for each layer (parallel to
 * hidden_layers)
 * @field output           Final output tensor from most recent forward pass
 * @field arena            Memory arena for layer/array allocations
 */
struct DenseContext {
  TensorContext *tensor_ctx;
  size_t input_size;
  DenseArr *hidden_layers;
  TensorArr *layer_inputs;
  TensorArr *pre_activations;
  OptimizerStateArr *optimizer_state;
  Tensor *output;
  Arena *arena;
};

DenseContext *dense_init(size_t in_features) {
  if (in_features == 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "in_features must be positive (received: %zu)",
                         in_features);
    return NULL;
  }

  Arena *arena = arena_create(dense_ARENA_SIZE);
  if (!arena) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
        "failed to allocate dense context (requested size: %zu bytes)",
        dense_ARENA_SIZE);
    return NULL;
  }

  DenseContext *ctx = calloc(1, sizeof(DenseContext));
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to allocate dense context");
    arena_free(arena);
    return NULL;
  }

  ctx->tensor_ctx = tensor_init();
  if (!ctx->tensor_ctx) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to create tensor context");
    arena_free(arena);
    free(ctx);
    return NULL;
  }

  DenseArr *dense_arr =
      arena_alloc(arena, sizeof(DenseArr), ARENA_ALIGNOF(DenseArr));
  if (!dense_arr) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to allocate dense array structure");
    tensor_free(ctx->tensor_ctx);
    arena_free(arena);
    free(ctx);
    return NULL;
  }

  Activation *activation =
      arena_alloc(arena, sizeof(Activation), ARENA_ALIGNOF(Activation));
  if (!activation) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to allocate activation array");
    tensor_free(ctx->tensor_ctx);
    arena_free(arena);
    free(ctx);
    return NULL;
  }

  dense_arr->capacity = 0;
  dense_arr->size = 0;
  dense_arr->items = NULL;

  TensorArr *layer_inputs =
      arena_alloc(arena, sizeof(TensorArr), ARENA_ALIGNOF(TensorArr));
  if (!layer_inputs) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to allocate layer inputs array");
    tensor_free(ctx->tensor_ctx);
    arena_free(arena);
    free(ctx);
    return NULL;
  }
  layer_inputs->capacity = 0;
  layer_inputs->size = 0;
  layer_inputs->items = NULL;

  TensorArr *pre_activations =
      arena_alloc(arena, sizeof(TensorArr), ARENA_ALIGNOF(TensorArr));
  if (!pre_activations) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to allocate pre-activations array");
    tensor_free(ctx->tensor_ctx);
    arena_free(arena);
    free(ctx);
    return NULL;
  }

  pre_activations->capacity = 0;
  pre_activations->size = 0;
  pre_activations->items = NULL;

  OptimizerStateArr *optimizer_state = arena_alloc(
      arena, sizeof(OptimizerStateArr), ARENA_ALIGNOF(OptimizerStateArr));
  if (!optimizer_state) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to allocate momentum states array");
    tensor_free(ctx->tensor_ctx);
    arena_free(arena);
    free(ctx);
    return NULL;
  }
  optimizer_state->capacity = 0;
  optimizer_state->size = 0;
  optimizer_state->items = NULL;

  ctx->output = NULL;
  ctx->input_size = in_features;
  ctx->hidden_layers = dense_arr;
  ctx->layer_inputs = layer_inputs;
  ctx->pre_activations = pre_activations;
  ctx->optimizer_state = optimizer_state;
  ctx->arena = arena;
  return ctx;
}

int dense_create(DenseContext *ctx, size_t out_features,
                 Activation activation) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "dense context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!ctx->tensor_ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (out_features == 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "out_features must be positive (received: %zu)",
                         out_features);
    return CTORCH_ERROR_INVALID_SHAPE;
  }

  Dense *layer = arena_alloc(ctx->arena, sizeof(Dense), ARENA_ALIGNOF(Dense));
  if (!layer) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
        "failed to allocate layer structure (requested size: %zu bytes)",
        sizeof(Dense));
    return CTORCH_ERROR_ARENA_ALLOCATION_FAILED;
  }

  switch (activation) {
  case ReLU:
    layer->weight =
        tensor_randn_he(ctx->tensor_ctx, ctx->input_size, out_features);
    break;
  default:
    // Sigmoid, Tanh, Linear, Softmax all use Xavier initialization
    layer->weight =
        tensor_randn_xavier(ctx->tensor_ctx, ctx->input_size, out_features);
    break;
  }

  if (!layer->weight)
    return CTORCH_ERROR_INVALID;

  layer->bias = tensor_zeros(ctx->tensor_ctx, 1, out_features);
  if (!layer->bias)
    return CTORCH_ERROR_INVALID;

  layer->activation = activation;

  OptimizerState *m_state =
      optimizer_state_init(ctx, ctx->input_size, out_features);
  if (!m_state)
    return CTORCH_ERROR_ARENA_ALLOCATION_FAILED;

  array_append(ctx->arena, ctx->optimizer_state, m_state);
  array_append(ctx->arena, ctx->hidden_layers, layer);
  ctx->input_size = out_features;
  return 0;
}

Tensor *dense_forward(DenseContext *ctx, Tensor *input) {
  if (!ctx) {
    return NULL;
  }

  if (!ctx->tensor_ctx) {
    return NULL;
  }

  if (!ctx->hidden_layers) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "dense layer is NULL");
    return NULL;
  }

  if (!input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return NULL;
  }

  // Reset caches for new forward pass
  ctx->layer_inputs->size = 0;
  ctx->pre_activations->size = 0;

  Tensor *layer_input = input;
  Tensor *output = NULL;

  for (size_t i = 0; i < ctx->hidden_layers->size; i++) {
    Dense *layer = ctx->hidden_layers->items[i];

    // Cache this layer's input for backward pass
    array_append(ctx->arena, ctx->layer_inputs, layer_input);

    float *bias_values = tensor_slice(ctx->tensor_ctx, layer->bias, 0, AxisRow);
    if (!bias_values) {
      ctorch_set_error(CTORCH_ERROR_NULL_DATA, "failed to extract bias values");
      return NULL;
    }

    output = linear(ctx->tensor_ctx, layer_input, layer->weight, bias_values);
    if (!output) {
      return NULL;
    }

    // Cache layer output for backward pass (activation functions modify
    // in-place, so after forward pass this will contain post-activation values)
    array_append(ctx->arena, ctx->pre_activations, output);

    switch (layer->activation) {
    case ReLU:
      relu(output);
      break;
    case Sigmoid:
      sigmoid(output);
      break;
    case Softmax:
      softmax(output);
      break;
    case Tanh:
      tanh_(output);
      break;
    default:
      break;
    }

    layer_input = output;

    // Cache final output
    ctx->output = output;
  }

  return output;
}

void dense_free(DenseContext *ctx) {
  if (!ctx)
    return;

  if (ctx->tensor_ctx)
    tensor_free(ctx->tensor_ctx);

  if (ctx->arena)
    arena_free(ctx->arena);

  free(ctx);
}

int sgd(float lr, Dense *layer, Tensor *dw, Tensor *db) {
  if (!layer || !dw || !db) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "layer, dw, or db is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (dw->rows != layer->weight->rows || dw->cols != layer->weight->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH, "dw shape mismatch");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  float *w = layer->weight->data;
  float *gW = dw->data;

  size_t size = layer->weight->rows * layer->weight->cols;
  for (size_t i = 0; i < size; i++)
    w[i] -= lr * gW[i];

  for (size_t i = 0; i < layer->bias->cols; i++)
    layer->bias->data[i] -= lr * db->data[i];

  return 0;
}

OptimizerState *optimizer_state_init(DenseContext *ctx, size_t in_features,
                                     size_t out_features) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (in_features == 0 || out_features == 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "dimensions must be positive (received: %zux%zu)",
                         in_features, out_features);
    return NULL;
  }

  OptimizerState *state = arena_alloc(ctx->arena, sizeof(OptimizerState),
                                      ARENA_ALIGNOF(OptimizerState));
  if (!state) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to allocate optimizer state");
    return NULL;
  }

  // v_weights shape: (in_features × out_features) - matches layer->weight
  state->v_weights = tensor_zeros(ctx->tensor_ctx, in_features, out_features);
  if (!state->v_weights) {
    free(state);
    return NULL;
  }

  // v_biases shape: (1 × out_features) - matches layer->bias
  state->v_biases = tensor_zeros(ctx->tensor_ctx, 1, out_features);
  if (!state->v_biases) {
    free(state);
    return NULL;
  }

  state->m_weights = tensor_zeros(ctx->tensor_ctx, in_features, out_features);
  if (!state->m_weights) {
    free(state);
    return NULL;
  }

  state->m_biases = tensor_zeros(ctx->tensor_ctx, 1, out_features);
  if (!state->m_biases) {
    free(state);
    return NULL;
  }

  state->t = 0;
  state->beta1_correction = 1.0f;
  state->beta2_correction = 1.0f;

  return state;
}

int momentum(OptimizerState *momentum_state, float lr, Dense *layer, Tensor *dw,
             Tensor *db) {
  if (!momentum_state || !layer || !dw || !db) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, momentum_state
                                                      ? "Momentum state is NULL"
                                                  : layer ? "Layer is NULL"
                                                  : dw    ? "dw is NULL"
                                                  : db    ? "db is NULL"
                                                          : "Unknown error");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (dw->rows != layer->weight->rows || dw->cols != layer->weight->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH, "dw shape mismatch");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  float beta = MOMENTUM_BETA;
  float beta_t = 1.0f - beta;

  size_t w_size = layer->weight->rows * layer->weight->cols;
  for (size_t i = 0; i < w_size; i++) {
    float g = dw->data[i];

    momentum_state->v_weights->data[i] =
        beta * momentum_state->v_weights->data[i] + beta_t * g;

    layer->weight->data[i] -= lr * momentum_state->v_weights->data[i];
  }

  for (size_t i = 0; i < layer->bias->cols; i++) {
    float g = db->data[i];

    momentum_state->v_biases->data[i] =
        beta * momentum_state->v_biases->data[i] + beta_t * g;

    layer->bias->data[i] -= lr * momentum_state->v_biases->data[i];
  }

  return 0;
}

int rmsprop(OptimizerState *optimizer_state, float lr, Dense *layer, Tensor *dw,
            Tensor *db) {
  if (!optimizer_state || !layer || !dw || !db) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     optimizer_state ? "Optimizer state is NULL"
                     : layer         ? "Layer is NULL"
                     : dw            ? "dw is NULL"
                     : db            ? "db is NULL"
                                     : "Unknown error");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (dw->rows != layer->weight->rows || dw->cols != layer->weight->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH, "dw shape mismatch");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  float beta = RMSPROP_BETA;
  float eps = EPSILON;

  size_t w_size = layer->weight->rows * layer->weight->cols;
  for (size_t i = 0; i < w_size; i++) {
    float g = dw->data[i];

    optimizer_state->v_weights->data[i] =
        beta * optimizer_state->v_weights->data[i] + (1 - beta) * g * g;

    layer->weight->data[i] -=
        lr * g / (sqrtf(optimizer_state->v_weights->data[i]) + eps);
  }

  for (size_t i = 0; i < layer->bias->cols; i++) {
    float g = db->data[i];

    optimizer_state->v_biases->data[i] =
        beta * optimizer_state->v_biases->data[i] + (1 - beta) * g * g;

    layer->bias->data[i] -=
        lr * g / (sqrtf(optimizer_state->v_biases->data[i]) + eps);
  }

  return 0;
}

int adam(OptimizerState *optimizer_state, float lr, Dense *layer, Tensor *dw,
         Tensor *db) {
  if (!optimizer_state || !layer || !dw || !db) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     optimizer_state ? "Optimizer state is NULL"
                     : layer         ? "Layer is NULL"
                     : dw            ? "dw is NULL"
                     : db            ? "db is NULL"
                                     : "Unknown error");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (dw->rows != layer->weight->rows || dw->cols != layer->weight->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH, "dw shape mismatch");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  optimizer_state->t++;

  float beta1 = ADAM_BETA1;
  float beta2 = ADAM_BETA2;
  float eps = EPSILON;

  optimizer_state->beta1_correction *= beta1;
  optimizer_state->beta2_correction *= beta2;

  float beta1_t = optimizer_state->beta1_correction;
  float beta2_t = optimizer_state->beta2_correction;

  size_t w_size = layer->weight->rows * layer->weight->cols;
  for (size_t i = 0; i < w_size; i++) {
    float g = dw->data[i];

    optimizer_state->m_weights->data[i] =
        beta1 * optimizer_state->m_weights->data[i] + (1 - beta1) * g;

    optimizer_state->v_weights->data[i] =
        beta2 * optimizer_state->v_weights->data[i] + (1 - beta2) * g * g;

    float m_hat = optimizer_state->m_weights->data[i] / (1 - beta1_t);
    float v_hat = optimizer_state->v_weights->data[i] / (1 - beta2_t);

    layer->weight->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }

  for (size_t i = 0; i < layer->bias->cols; i++) {
    float g = db->data[i];
    optimizer_state->m_biases->data[i] =
        beta1 * optimizer_state->m_biases->data[i] + (1 - beta1) * g;

    optimizer_state->v_biases->data[i] =
        beta2 * optimizer_state->v_biases->data[i] + (1 - beta2) * g * g;

    float m_hat = optimizer_state->m_biases->data[i] / (1 - beta1_t);
    float v_hat = optimizer_state->v_biases->data[i] / (1 - beta2_t);

    layer->bias->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }

  return 0;
}

int dense_backward(DenseContext *ctx, Tensor *grad_output, float lr,
                   OptimizerType optimizer) {
  if (!ctx || !grad_output) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     "context or grad_output is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!ctx->tensor_ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!ctx->hidden_layers || ctx->hidden_layers->size == 0) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "no layers in context");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!grad_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "grad_output data is NULL");
    return CTORCH_ERROR_NULL_DATA;
  }

  if (ctx->layer_inputs->size != ctx->hidden_layers->size) {
    ctorch_set_error(CTORCH_ERROR_INVALID,
                     "forward pass must be called before backward pass");
    return CTORCH_ERROR_INVALID;
  }

  Tensor *upstream = grad_output;

  for (size_t idx = ctx->hidden_layers->size; idx > 0; idx--) {
    size_t i = idx - 1;
    Dense *layer = ctx->hidden_layers->items[i];
    Tensor *layer_input = ctx->layer_inputs->items[i];
    Tensor *layer_output = ctx->pre_activations->items[i];

    // For Softmax with cross-entropy, the combined gradient is already computed
    // by cross_entropy_backward, so we skip activation backward for Softmax
    Tensor *activation_grad = upstream;
    if (layer->activation != Softmax) {
      switch (layer->activation) {
      case ReLU:
        activation_grad =
            relu_backward(ctx->tensor_ctx, upstream, layer_output);
        break;
      case Sigmoid:
        activation_grad =
            sigmoid_backward(ctx->tensor_ctx, upstream, layer_output);
        break;
      case Tanh:
        activation_grad =
            tanh_backward(ctx->tensor_ctx, upstream, layer_output);
        break;
      default:
        break;
      }
      if (!activation_grad) {
        return CTORCH_ERROR_INVALID;
      }
    }

    Tensor *dw = weight_gradient(ctx->tensor_ctx, layer_input, activation_grad);
    if (!dw) {
      return CTORCH_ERROR_INVALID;
    }

    Tensor *db = bias_gradient(ctx->tensor_ctx, activation_grad);
    if (!db) {
      return CTORCH_ERROR_INVALID;
    }

    Tensor *dx =
        input_gradient(ctx->tensor_ctx, activation_grad, layer->weight);
    if (!dx) {
      return CTORCH_ERROR_INVALID;
    }

    switch (optimizer) {
    case Momentum:
      momentum(ctx->optimizer_state->items[i], lr, layer, dw, db);
      break;
    case SGD:
      sgd(lr, layer, dw, db);
      break;
    case RMSprop:
      rmsprop(ctx->optimizer_state->items[i], lr, layer, dw, db);
      break;
    case Adam:
      adam(ctx->optimizer_state->items[i], lr, layer, dw, db);
      break;
    default:
      break;
    }

    upstream = dx;
  }

  return 0;
}

Tensor *predict(DenseContext *ctx, Tensor *input) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!ctx->tensor_ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor context is NULL");
    return NULL;
  }

  if (!input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return NULL;
  }

  Tensor *output = dense_forward(ctx, input);
  if (!output) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "predict failed");
    return NULL;
  }

  Tensor *predictions = tensor_create(ctx->tensor_ctx, 1);
  if (!predictions) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "predict failed");
    return NULL;
  }

  for (size_t i = 0; i < output->rows; i++) {
    float pred_idx[1] = {0.0f};
    float max_prob = -1.0f;
    for (size_t j = 0; j < output->cols; j++) {
      float prob = output->data[i * output->cols + j];
      if (prob > max_prob) {
        max_prob = prob;
        pred_idx[0] = (float)j;
      }
    }
    tensor_append(ctx->tensor_ctx, predictions, pred_idx);
  }

  return predictions;
}

float accuracy(DenseContext *ctx, Tensor *predictions, Tensor *targets) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NAN;
  }

  if (!ctx->tensor_ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor context is NULL");
    return NAN;
  }

  if (!predictions) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "predictions tensor is NULL");
    return NAN;
  }

  if (!targets) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "targets tensor is NULL");
    return NAN;
  }

  if (predictions->rows != targets->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "predictions and targets must have the "
                         "same number of rows (received: %zux%zu vs %zux%zu)",
                         predictions->rows, predictions->cols, targets->rows,
                         targets->cols);
    return NAN;
  }

  int correct = 0;
  bool is_one_hot = targets->cols > 1;

  for (size_t i = 0; i < predictions->rows; i++) {
    float pred = predictions->data[i * predictions->cols];
    float target_class;

    if (is_one_hot) {
      float max_val = -1.0f;
      target_class = 0.0f;
      for (size_t j = 0; j < targets->cols; j++) {
        float val = targets->data[i * targets->cols + j];
        if (val > max_val) {
          max_val = val;
          target_class = (float)j;
        }
      }
    } else {
      target_class = targets->data[i * targets->cols];
    }

    if (pred == target_class)
      correct++;
  }

  return (float)correct / (float)predictions->rows;
}
