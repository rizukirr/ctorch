#include "keras.h"
#include "arena.h"
#include "errors.h"
#include "ops.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define dense_ARENA_SIZE 1024 * 16

typedef struct {
  size_t capacity;
  size_t size;
  Dense **items;
} DenseArr;

typedef struct {
  size_t capacity;
  size_t size;
  Tensor **items;
} TensorArr;

struct OptimizerState {
  Tensor *v_weights;
  Tensor *v_biases;
  Tensor *m_weights;
  Tensor *m_biases;
  int t;
};

typedef struct {
  size_t capacity;
  size_t size;
  OptimizerState **items;
} OptimizerStateArr;

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

  // Allocate the DenseArr structure for layer tracking
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

  // Allocate arrays for caching intermediate values during forward pass
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
  case Sigmoid:
  case Tanh:
    layer->weight =
        tensor_randn_xavier(ctx->tensor_ctx, ctx->input_size, out_features);
    break;
  case Linear:
    layer->weight =
        tensor_randn_xavier(ctx->tensor_ctx, ctx->input_size, out_features);
    break;
  default:
    layer->weight =
        tensor_randn(ctx->tensor_ctx, ctx->input_size, out_features);
    break;
  }

  if (!layer->weight)
    return CTORCH_ERROR_INVALID;

  layer->bias = tensor_zeros(ctx->tensor_ctx, 1, out_features);
  if (!layer->bias)
    return CTORCH_ERROR_INVALID;

  layer->activation = activation;

  // Create momentum state for this layer
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

    // Get the bias row (all bias values for this layer)
    float *bias_values = tensor_slice(ctx->tensor_ctx, layer->bias, 0, AxisRow);
    if (!bias_values) {
      ctorch_set_error(CTORCH_ERROR_NULL_DATA, "failed to extract bias values");
      return NULL;
    }

    // Apply linear transformation for this layer
    output = linear(ctx->tensor_ctx, layer_input, layer->weight, bias_values);
    if (!output) {
      return NULL;
    }

    // Cache layer output for backward pass (activation functions modify
    // in-place, so after forward pass this will contain post-activation values)
    array_append(ctx->arena, ctx->pre_activations, output);

    // Apply this layer's activation function
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

  float beta = 0.9f;
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

  float beta = 0.999f;
  float eps = 1e-8f;

  size_t w_size = layer->weight->rows * layer->weight->cols;
  for (size_t i = 0; i < w_size; i++) {
    float g = dw->data[i];

    optimizer_state->v_weights->data[i] =
        beta * optimizer_state->v_weights->data[i] + (1 - beta) * powf(g, 2);

    layer->weight->data[i] -=
        lr * g / (sqrtf(optimizer_state->v_weights->data[i]) + eps);
  }

  for (size_t i = 0; i < layer->bias->cols; i++) {
    float g = db->data[i];

    optimizer_state->v_biases->data[i] =
        beta * optimizer_state->v_biases->data[i] + (1 - beta) * powf(g, 2);

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

  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;

  int t = optimizer_state->t;
  float beta1_t = powf(beta1, t);
  float beta2_t = powf(beta2, t);

  size_t w_size = layer->weight->rows * layer->weight->cols;
  for (size_t i = 0; i < w_size; i++) {
    float g = dw->data[i];

    optimizer_state->m_weights->data[i] =
        beta1 * optimizer_state->m_weights->data[i] + (1 - beta1) * g;

    optimizer_state->v_weights->data[i] =
        beta2 * optimizer_state->v_weights->data[i] + (1 - beta2) * powf(g, 2);

    float m_hat = optimizer_state->m_weights->data[i] / (1 - beta1_t);
    float v_hat = optimizer_state->v_weights->data[i] / (1 - beta2_t);

    layer->weight->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }

  for (size_t i = 0; i < layer->bias->cols; i++) {
    float g = db->data[i];
    optimizer_state->m_biases->data[i] =
        beta1 * optimizer_state->m_biases->data[i] + (1 - beta1) * g;

    optimizer_state->v_biases->data[i] =
        beta2 * optimizer_state->v_biases->data[i] + (1 - beta2) * powf(g, 2);

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

  // Current gradient flowing backward (dL/dY for current layer)
  Tensor *upstream = grad_output;

  // Iterate from last layer to first (backpropagation)
  for (size_t idx = ctx->hidden_layers->size; idx > 0; idx--) {
    size_t i = idx - 1; // Current layer index (0-based)
    Dense *layer = ctx->hidden_layers->items[i];
    Tensor *layer_input = ctx->layer_inputs->items[i];
    Tensor *layer_output =
        ctx->pre_activations->items[i]; // Post-activation values

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

    // Compute weight gradient: dL/dW = input^T @ activation_grad
    Tensor *dw = weight_gradient(ctx->tensor_ctx, layer_input, activation_grad);
    if (!dw) {
      return CTORCH_ERROR_INVALID;
    }

    // Compute bias gradient: dL/db = sum(activation_grad, axis=0)
    Tensor *db = bias_gradient(ctx->tensor_ctx, activation_grad);
    if (!db) {
      return CTORCH_ERROR_INVALID;
    }

    // Compute input gradient: dL/dX = activation_grad @ weight^T
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

    // Propagate gradient to previous layer
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
      float prob = tensor_get(output, i, j);
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
    float pred = tensor_get(predictions, i, 0);
    float target_class;

    if (is_one_hot) {
      // Find argmax of one-hot encoded row
      float max_val = -1.0f;
      target_class = 0.0f;
      for (size_t j = 0; j < targets->cols; j++) {
        float val = tensor_get(targets, i, j);
        if (val > max_val) {
          max_val = val;
          target_class = (float)j;
        }
      }
    } else {
      target_class = tensor_get(targets, i, 0);
    }

    if (pred == target_class)
      correct++;
  }

  return (float)correct / (float)predictions->rows;
}
