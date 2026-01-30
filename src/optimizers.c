#include "optimizers.h"
#include "errors.h"
#include <math.h>
#include <stdlib.h>

OptimizerState *optimizer_state_init(TensorContext *ctx, size_t in_features,
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

  OptimizerState *state = calloc(1, sizeof(OptimizerState));
  if (!state) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate optimizer state");
    return NULL;
  }

  // v_weights shape: (in_features x out_features) - matches layer->weight
  state->v_weights = tensor_zeros(ctx, in_features, out_features);
  if (!state->v_weights) {
    free(state);
    return NULL;
  }

  // v_biases shape: (1 x out_features) - matches layer->bias
  state->v_biases = tensor_zeros(ctx, 1, out_features);
  if (!state->v_biases) {
    free(state);
    return NULL;
  }

  state->m_weights = tensor_zeros(ctx, in_features, out_features);
  if (!state->m_weights) {
    free(state);
    return NULL;
  }

  state->m_biases = tensor_zeros(ctx, 1, out_features);
  if (!state->m_biases) {
    free(state);
    return NULL;
  }

  state->t = 0;
  state->beta1_correction = 1.0f;
  state->beta2_correction = 1.0f;

  return state;
}

int sgd(float lr, Tensor *weight, Tensor *bias, Tensor *dw, Tensor *db) {
  if (!weight || !bias || !dw || !db) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     "weight, bias, dw, or db is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (dw->rows != weight->rows || dw->cols != weight->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH, "dw shape mismatch");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  float *w = weight->data;
  float *gW = dw->data;

  size_t size = weight->rows * weight->cols;
  for (size_t i = 0; i < size; i++)
    w[i] -= lr * gW[i];

  for (size_t i = 0; i < bias->cols; i++)
    bias->data[i] -= lr * db->data[i];

  return 0;
}

int momentum(OptimizerState *state, float lr, Tensor *weight, Tensor *bias,
             Tensor *dw, Tensor *db) {
  if (!state || !weight || !bias || !dw || !db) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !state    ? "optimizer state is NULL"
                     : !weight ? "weight is NULL"
                     : !bias   ? "bias is NULL"
                     : !dw     ? "dw is NULL"
                               : "db is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (dw->rows != weight->rows || dw->cols != weight->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH, "dw shape mismatch");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  float beta = MOMENTUM_BETA;
  float beta_t = 1.0f - beta;

  size_t w_size = weight->rows * weight->cols;
  for (size_t i = 0; i < w_size; i++) {
    float g = dw->data[i];

    state->v_weights->data[i] = beta * state->v_weights->data[i] + beta_t * g;

    weight->data[i] -= lr * state->v_weights->data[i];
  }

  for (size_t i = 0; i < bias->cols; i++) {
    float g = db->data[i];

    state->v_biases->data[i] = beta * state->v_biases->data[i] + beta_t * g;

    bias->data[i] -= lr * state->v_biases->data[i];
  }

  return 0;
}

int rmsprop(OptimizerState *state, float lr, Tensor *weight, Tensor *bias,
            Tensor *dw, Tensor *db) {
  if (!state || !weight || !bias || !dw || !db) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !state    ? "optimizer state is NULL"
                     : !weight ? "weight is NULL"
                     : !bias   ? "bias is NULL"
                     : !dw     ? "dw is NULL"
                               : "db is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (dw->rows != weight->rows || dw->cols != weight->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH, "dw shape mismatch");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  float beta = RMSPROP_BETA;
  float eps = EPSILON;

  size_t w_size = weight->rows * weight->cols;
  for (size_t i = 0; i < w_size; i++) {
    float g = dw->data[i];

    state->v_weights->data[i] =
        beta * state->v_weights->data[i] + (1 - beta) * g * g;

    weight->data[i] -= lr * g / (sqrtf(state->v_weights->data[i]) + eps);
  }

  for (size_t i = 0; i < bias->cols; i++) {
    float g = db->data[i];

    state->v_biases->data[i] =
        beta * state->v_biases->data[i] + (1 - beta) * g * g;

    bias->data[i] -= lr * g / (sqrtf(state->v_biases->data[i]) + eps);
  }

  return 0;
}

int adam(OptimizerState *state, float lr, Tensor *weight, Tensor *bias,
         Tensor *dw, Tensor *db) {
  if (!state || !weight || !bias || !dw || !db) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !state    ? "optimizer state is NULL"
                     : !weight ? "weight is NULL"
                     : !bias   ? "bias is NULL"
                     : !dw     ? "dw is NULL"
                               : "db is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (dw->rows != weight->rows || dw->cols != weight->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH, "dw shape mismatch");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  state->t++;

  float beta1 = ADAM_BETA1;
  float beta2 = ADAM_BETA2;
  float eps = EPSILON;

  state->beta1_correction *= beta1;
  state->beta2_correction *= beta2;

  float beta1_t = state->beta1_correction;
  float beta2_t = state->beta2_correction;

  size_t w_size = weight->rows * weight->cols;
  for (size_t i = 0; i < w_size; i++) {
    float g = dw->data[i];

    state->m_weights->data[i] =
        beta1 * state->m_weights->data[i] + (1 - beta1) * g;

    state->v_weights->data[i] =
        beta2 * state->v_weights->data[i] + (1 - beta2) * g * g;

    float m_hat = state->m_weights->data[i] / (1 - beta1_t);
    float v_hat = state->v_weights->data[i] / (1 - beta2_t);

    weight->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }

  for (size_t i = 0; i < bias->cols; i++) {
    float g = db->data[i];
    state->m_biases->data[i] =
        beta1 * state->m_biases->data[i] + (1 - beta1) * g;

    state->v_biases->data[i] =
        beta2 * state->v_biases->data[i] + (1 - beta2) * g * g;

    float m_hat = state->m_biases->data[i] / (1 - beta1_t);
    float v_hat = state->v_biases->data[i] / (1 - beta2_t);

    bias->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }

  return 0;
}
