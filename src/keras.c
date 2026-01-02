#include "keras.h"
#include "arena.h"
#include "errors.h"
#include "ops.h"
#include <stdio.h>
#include <stdlib.h>

#define dense_ARENA_SIZE 1024 * 16

struct DenseContext {
  TensorContext *tensor_ctx;
  size_t input_size;
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
  ctx->input_size = input_size;
  ctx->arena = arena;
  return ctx;
}

Dense *dense_create(DenseContext *ctx, size_t output_size) {
  if (!ctx) {
    return NULL;
  }

  if (!ctx->tensor_ctx) {
    return NULL;
  }

  if (output_size == 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "layer dimensions must be positive (output_size: %zu)",
                         output_size);
    return NULL;
  }

  Dense *dense = arena_alloc(ctx->arena, sizeof(Dense), ARENA_ALIGNOF(Dense));
  if (!dense) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
        "failed to allocate layer structure (requested size: %zu bytes)",
        sizeof(Dense));
    return NULL;
  }

  dense->weights = tensor_randn(ctx->tensor_ctx, ctx->input_size, output_size);
  if (!dense->weights)
    return NULL;

  dense->biases = scalar_randn(ctx->tensor_ctx, output_size);
  if (!dense->biases)
    return NULL;

  dense->grad_biases = NULL;
  dense->outputs = NULL;
  dense->inputs = NULL;
  dense->pre_activation = NULL;
  dense->output_size = output_size;

  ctx->input_size = output_size;
  return dense;
}

Tensor *dense_forward(DenseContext *ctx, Dense *dense, Tensor *inputs,
                      Activation activation) {
  if (!ctx) {
    return NULL;
  }

  if (!ctx->tensor_ctx) {
    return NULL;
  }

  if (!dense) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "dense layer is NULL");
    return NULL;
  }

  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return NULL;
  }

  if (!dense->weights || !dense->biases) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA,
                     "dense layer not properly initialized");
    return NULL;
  }

  // Cache input for backward pass
  dense->inputs = inputs;

  // Compute affine transformation
  Tensor *af =
      affine_transform(ctx->tensor_ctx, inputs, dense->weights, dense->biases);

  if (!af) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "affine transform failed");
    return NULL;
  }

  // Cache pre-activation for backward pass
  dense->pre_activation = tensor_copy(ctx->tensor_ctx, af);
  if (!dense->pre_activation) {
    return NULL;
  }

  // Cache activation type for backward pass
  dense->activation = activation;

  // Apply activation function in-place
  int ret = 0;
  switch (activation) {
  case None:
    // No activation
    break;
  case ReLU:
    ret = relu(af);
    break;
  case Sigmoid:
    ret = sigmoid(af);
    break;
  case Softmax:
    ret = softmax(af);
    break;
  case Tanh:
    ret = tanhh(af);
    break;
  default:
    break;
  }

  if (ret < 0) {
    return NULL;
  }

  // Cache output (post-activation) for backward pass
  dense->outputs = af;

  return af;
}

Tensor *dense_backward(DenseContext *ctx, Dense *dense, Tensor *grad_output) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!dense) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "dense layer is NULL");
    return NULL;
  }

  if (!grad_output) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     "gradient output tensor is NULL");
    return NULL;
  }

  // Allocate gradient for pre-activation tensor
  Tensor *grad_pre_activation = tensor_new(ctx->tensor_ctx, grad_output->cols);
  if (!grad_pre_activation)
    return NULL;

  // Copy shape from grad_output
  for (size_t i = 0; i < grad_output->rows; i++) {
    float row[grad_output->cols];
    for (size_t j = 0; j < grad_output->cols; j++) {
      row[j] = 0.0f;
    }
    tensor_append(ctx->tensor_ctx, grad_pre_activation, row);
  }

  // Backward through activation function
  int ret = 0;
  switch (dense->activation) {
  case None:
    // No activation - pass through gradient
    memcpy(grad_pre_activation->data, grad_output->data,
           grad_output->rows * grad_output->cols * sizeof(float));
    break;
  case ReLU:
    ret = relu_backward(ctx->tensor_ctx, grad_output, dense->pre_activation,
                        grad_pre_activation);
    break;
  case Sigmoid:
    ret = sigmoid_backward(ctx->tensor_ctx, grad_output, dense->outputs,
                           grad_pre_activation);
    break;
  case Tanh:
    ret = tanh_backward(ctx->tensor_ctx, grad_output, dense->outputs,
                        grad_pre_activation);
    break;
  case Softmax:
    // For softmax, gradient typically comes from combined softmax+cross_entropy
    // In this case, just pass through the gradient
    memcpy(grad_pre_activation->data, grad_output->data,
           grad_output->rows * grad_output->cols * sizeof(float));
    break;
  default:
    // No activation - pass through gradient
    memcpy(grad_pre_activation->data, grad_output->data,
           grad_output->rows * grad_output->cols * sizeof(float));
    break;
  }

  if (ret < 0) {
    return NULL;
  }

  // Backward through affine transformation
  Tensor *grad_inputs = NULL;
  Tensor *grad_weights = NULL;
  float *grad_bias = NULL;

  affine_backward(ctx->tensor_ctx, grad_pre_activation, dense->inputs,
                  dense->weights, &grad_inputs, &grad_weights, &grad_bias);

  if (!grad_inputs || !grad_weights || !grad_bias) {
    return NULL;
  }

  // Allocate and accumulate gradients for weights
  tensor_allocate_grad(ctx->tensor_ctx, dense->weights);
  if (dense->weights->grad) {
    // Accumulate weight gradients
    for (size_t i = 0; i < dense->weights->rows * dense->weights->cols; i++) {
      dense->weights->grad[i] += grad_weights->data[i];
    }
  }

  // Allocate and accumulate gradients for biases
  if (!dense->grad_biases) {
    dense->grad_biases = arena_alloc(
        ctx->arena, dense->output_size * sizeof(float), ARENA_ALIGNOF(float));
    if (!dense->grad_biases) {
      ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                       "failed to allocate bias gradients");
      return NULL;
    }
    memset(dense->grad_biases, 0, dense->output_size * sizeof(float));
  }

  // Accumulate bias gradients
  for (size_t i = 0; i < dense->output_size; i++) {
    dense->grad_biases[i] += grad_bias[i];
  }

  return grad_inputs;
}

void dense_zero_grad(Dense *dense) {
  if (!dense)
    return;

  // Zero weight gradients
  if (dense->weights && dense->weights->grad) {
    tensor_zero_grad(dense->weights);
  }

  // Zero bias gradients
  if (dense->grad_biases) {
    memset(dense->grad_biases, 0, dense->output_size * sizeof(float));
  }
}

int sgd_step(DenseContext *ctx, Dense **layers, size_t num_layers,
             float learning_rate) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!layers) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "layers array is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (learning_rate <= 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "learning rate must be positive (received: %f)",
                         learning_rate);
    return CTORCH_ERROR_INVALID_SHAPE;
  }

  // Update each layer
  for (size_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
    Dense *layer = layers[layer_idx];
    if (!layer)
      continue;

    // Update weights: weights -= learning_rate * grad_weights
    if (layer->weights && layer->weights->grad) {
      for (size_t i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
        layer->weights->data[i] -= learning_rate * layer->weights->grad[i];
      }
    }

    // Update biases: biases -= learning_rate * grad_biases
    if (layer->biases && layer->grad_biases) {
      for (size_t i = 0; i < layer->output_size; i++) {
        layer->biases[i] -= learning_rate * layer->grad_biases[i];
      }
    }
  }

  return 0;
}

int dense_free(DenseContext *ctx) {
  if (!ctx) {
    return 0; // NULL is valid for free operations
  }

  if (ctx->tensor_ctx)
    tensor_free(ctx->tensor_ctx);

  if (ctx->arena)
    arena_free(ctx->arena);

  free(ctx);
  return 0;
}
