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

struct DenseContext {
  TensorContext *tensor_ctx;
  size_t input_size;
  DenseArr *hidden_layers;
  TensorArr *layer_inputs; // Cached inputs to each layer (for weight_gradient)
  TensorArr *
      pre_activations; // Cached pre-activation values (for activation backward)
  Tensor *output;
  Arena *arena;
};

DenseContext *dense_init(size_t input_size) {
  if (input_size == 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "input size must be positive (received: %zu)",
                         input_size);
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

  ctx->tensor_ctx = tensor_create();
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

  ctx->output = NULL;
  ctx->input_size = input_size;
  ctx->hidden_layers = dense_arr;
  ctx->layer_inputs = layer_inputs;
  ctx->pre_activations = pre_activations;
  ctx->arena = arena;
  return ctx;
}

int dense_create(DenseContext *ctx, size_t output_size, Activation activation) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "dense context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!ctx->tensor_ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (output_size == 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "layer dimensions must be positive (output_size: %zu)",
                         output_size);
    return CTORCH_ERROR_INVALID_SHAPE;
  }

  Dense *dense = arena_alloc(ctx->arena, sizeof(Dense), ARENA_ALIGNOF(Dense));
  if (!dense) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
        "failed to allocate layer structure (requested size: %zu bytes)",
        sizeof(Dense));
    return CTORCH_ERROR_ARENA_ALLOCATION_FAILED;
  }

  dense->weights = tensor_randn(ctx->tensor_ctx, ctx->input_size, output_size);
  if (!dense->weights)
    return CTORCH_ERROR_INVALID;

  dense->biases = tensor_randn(ctx->tensor_ctx, 1, output_size);
  if (!dense->biases)
    return CTORCH_ERROR_INVALID;

  dense->activation = activation;
  ctx->input_size = output_size;
  array_append(ctx->arena, ctx->hidden_layers, dense);
  return 0;
}

Tensor *dense_forward(DenseContext *ctx, Tensor *inputs) {
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

  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return NULL;
  }

  // Reset caches for new forward pass
  ctx->layer_inputs->size = 0;
  ctx->pre_activations->size = 0;

  Tensor *layer_input = inputs;
  Tensor *af = NULL;

  for (size_t i = 0; i < ctx->hidden_layers->size; i++) {
    Dense *current_dense = ctx->hidden_layers->items[i];

    // Cache this layer's input for backward pass
    array_append(ctx->arena, ctx->layer_inputs, layer_input);

    // Get the bias row (all bias values for this layer)
    float *bias =
        tensor_slice(ctx->tensor_ctx, current_dense->biases, 0, AxisRow);
    if (!bias) {
      ctorch_set_error(CTORCH_ERROR_NULL_DATA, "failed to extract bias values");
      return NULL;
    }

    // Apply affine transformation for this layer
    af = affine_transform(ctx->tensor_ctx, layer_input, current_dense->weights,
                          bias);
    if (!af) {
      return NULL;
    }

    // Cache layer output for backward pass (activation functions modify
    // in-place, so after forward pass this will contain post-activation values)
    array_append(ctx->arena, ctx->pre_activations, af);

    // Apply this layer's activation function
    switch (current_dense->activation) {
    case ReLU:
      relu(af);
      break;
    case Sigmoid:
      sigmoid(af);
      break;
    case Softmax:
      softmax(af);
      break;
    case Tanh:
      tanhh(af);
      break;
    default:
      break;
    }

    layer_input = af;

    // Cache final output
    ctx->output = af;
  }

  return af;
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

int dense_backward(DenseContext *ctx, float learning_rate, Tensor *loss_grad) {
  if (!ctx || !loss_grad) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     "context or loss gradient is NULL");
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

  if (!loss_grad->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "loss gradient data is NULL");
    return CTORCH_ERROR_NULL_DATA;
  }

  if (ctx->layer_inputs->size != ctx->hidden_layers->size) {
    ctorch_set_error(CTORCH_ERROR_INVALID,
                     "forward pass must be called before backward pass");
    return CTORCH_ERROR_INVALID;
  }

  // Current gradient flowing backward (dL/dY for current layer)
  Tensor *upstream_grad = loss_grad;

  // Iterate from last layer to first (backpropagation)
  for (size_t idx = ctx->hidden_layers->size; idx > 0; idx--) {
    size_t i = idx - 1; // Current layer index (0-based)
    Dense *layer = ctx->hidden_layers->items[i];
    Tensor *layer_input = ctx->layer_inputs->items[i];
    Tensor *layer_output =
        ctx->pre_activations->items[i]; // Post-activation values

    // Step 1: Compute gradient through activation function
    // For Softmax with cross-entropy, the combined gradient is already computed
    // by cross_entropy_backward, so we skip activation backward for Softmax
    Tensor *activation_grad = upstream_grad;
    if (layer->activation != Softmax) {
      switch (layer->activation) {
      case ReLU:
        // relu_backward works with post-activation values since ReLU(x)>0 iff
        // x>0
        activation_grad =
            relu_backward(ctx->tensor_ctx, upstream_grad, layer_output);
        break;
      case Sigmoid:
        activation_grad =
            sigmoid_backward(ctx->tensor_ctx, upstream_grad, layer_output);
        break;
      case Tanh:
        activation_grad =
            tanh_backward(ctx->tensor_ctx, upstream_grad, layer_output);
        break;
      default:
        break;
      }
      if (!activation_grad) {
        return CTORCH_ERROR_INVALID;
      }
    }

    // Step 2: Compute weight gradient: dL/dW = X^T * dL/dZ
    Tensor *dw = weight_gradient(ctx->tensor_ctx, layer_input, activation_grad);
    if (!dw) {
      return CTORCH_ERROR_INVALID;
    }

    // Step 3: Compute bias gradient: dL/db = sum(dL/dZ, axis=0)
    Tensor *db = bias_gradient(ctx->tensor_ctx, activation_grad);
    if (!db) {
      return CTORCH_ERROR_INVALID;
    }

    // Step 4: Compute input gradient for next iteration: dL/dX = dL/dZ * W^T
    Tensor *dx =
        input_gradient(ctx->tensor_ctx, activation_grad, layer->weights);
    if (!dx) {
      return CTORCH_ERROR_INVALID;
    }

    // Step 5: Update weights: W = W - lr * dL/dW
    for (size_t j = 0; j < layer->weights->rows; j++) {
      for (size_t k = 0; k < layer->weights->cols; k++) {
        size_t weight_idx = j * layer->weights->cols + k;
        layer->weights->data[weight_idx] -=
            learning_rate * tensor_get(dw, j, k);
      }
    }

    // Step 6: Update biases: b = b - lr * dL/db
    for (size_t k = 0; k < layer->biases->cols; k++) {
      layer->biases->data[k] -= learning_rate * tensor_get(db, 0, k);
    }

    // Propagate gradient to previous layer
    upstream_grad = dx;
  }

  return 0;
}

Tensor *predict(DenseContext *ctx, Tensor *inputs) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!ctx->tensor_ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor context is NULL");
    return NULL;
  }

  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return NULL;
  }

  Tensor *output = dense_forward(ctx, inputs);
  if (!output) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "Predict failed");
    return NULL;
  }

  Tensor *p = tensor_new(ctx->tensor_ctx, 1);
  if (!p) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "Predict failed");
    return NULL;
  }

  for (size_t i = 0; i < output->rows; i++) {
    float pred_class[1] = {0.0f};
    float max_prob = -1.0f;
    for (size_t j = 0; j < output->cols; j++) {
      float prob = tensor_get(output, i, j);
      if (prob > max_prob) {
        max_prob = prob;
        pred_class[0] = (float)j;
      }
    }
    tensor_append(ctx->tensor_ctx, p, pred_class);
  }

  return p;
}

float accuracy(DenseContext *ctx, Tensor *pred_class, Tensor *true_class) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NAN;
  }

  if (!ctx->tensor_ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor context is NULL");
    return NAN;
  }

  if (!pred_class) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "prediction class is NULL");
    return NAN;
  }

  if (!true_class) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "true class is NULL");
    return NAN;
  }

  if (pred_class->rows != true_class->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "prediction and true class tensors must have the "
                         "same number of rows (received: %zux%zu vs %zux%zu)",
                         pred_class->rows, pred_class->cols, true_class->rows,
                         true_class->cols);
    return NAN;
  }

  int correct = 0;
  bool is_one_hot = true_class->cols > 1;

  for (size_t i = 0; i < pred_class->rows; i++) {
    float pred = tensor_get(pred_class, i, 0);
    float t;

    if (is_one_hot) {
      // Find argmax of one-hot encoded row
      float max_val = -1.0f;
      t = 0.0f;
      for (size_t j = 0; j < true_class->cols; j++) {
        float val = tensor_get(true_class, i, j);
        if (val > max_val) {
          max_val = val;
          t = (float)j;
        }
      }
    } else {
      t = tensor_get(true_class, i, 0);
    }

    if (pred == t)
      correct++;
  }

  return (float)correct / (float)pred_class->rows;
}
