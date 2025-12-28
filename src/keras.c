#include "keras.h"
#include "arena.h"
#include "errors.h"
#include "ops.h"
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

  dense->biases = tensor_randn(ctx->tensor_ctx, 1, output_size);
  if (!dense->biases)
    return NULL;

  dense->result = NULL;

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

  float *bias = tensor_slice(ctx->tensor_ctx, dense->biases, 0, AxisColum);
  if (!bias) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA,
                     "dense layer not properly initialized");
    return NULL;
  }

  Tensor *af = affine_transform(ctx->tensor_ctx, inputs, dense->weights, bias);

  switch (activation) {
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

  dense->result = af;
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
