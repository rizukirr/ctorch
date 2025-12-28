#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

Tensor *affine_transform(TensorContext *ctx, Tensor *inputs, Tensor *weights,
                         float *bias) {
  if (!ctx) {
    return NULL;
  }

  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return NULL;
  }

  if (!weights) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "weight tensor is NULL");
    return NULL;
  }

  if (!bias) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "bias vector is NULL");
    return NULL;
  }

  if (inputs->cols != weights->rows) {
    if (inputs->cols == weights->cols) {
      tensor_transpose(weights);
    } else {
      ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                           "dimension mismatch (inputs: %zux%zu, weights: "
                           "%zux%zu) - expected inputs.cols == weights.rows",
                           inputs->rows, inputs->cols, weights->rows,
                           weights->cols);
      return NULL;
    }
  }

  Tensor *outputs = tensor_new(ctx, weights->cols);
  if (!outputs)
    return NULL;

  for (size_t i = 0; i < inputs->rows; i++) {
    float dot[weights->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t k = 0; k < weights->cols; k++) {
      for (size_t j = 0; j < inputs->cols; j++) {
        float input = tensor_get(inputs, i, j);
        float weight = tensor_get(weights, j, k);
        dot[k] += input * weight;
      }
      dot[k] += bias[k];
    }
    tensor_append(ctx, outputs, dot);
  }
  return outputs;
}

void relu(Tensor *inputs) {
  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return;
  }

  Tensor *tmp = tensor_new_tmp(inputs->cols);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for ReLU");
    return;
  }

  for (size_t i = 0; i < inputs->rows; i++) {
    float dot[inputs->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = tensor_get(inputs, i, j);
      dot[j] = input > 0 ? input : 0;
    }
    tensor_append_tmp(tmp, dot);
  }

  memcpy(inputs->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  inputs->rows = tmp->rows;
  inputs->cols = tmp->cols;
  inputs->capacity = tmp->capacity;

  tensor_free_tmp(tmp);
}

void sigmoid(Tensor *inputs) {
  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return;
  }

  Tensor *tmp = tensor_new_tmp(inputs->cols);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for sigmoid");
    return;
  }

  for (size_t i = 0; i < inputs->rows; i++) {
    float dot[inputs->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = tensor_get(inputs, i, j);
      dot[j] = 1 / (1 + expf(-input));
    }
    tensor_append_tmp(tmp, dot);
  }

  memcpy(inputs->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  inputs->rows = tmp->rows;
  inputs->cols = tmp->cols;
  inputs->capacity = tmp->capacity;

  tensor_free_tmp(tmp);
}

void softmax(Tensor *inputs) {
  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return;
  }

  Tensor *tmp = tensor_new_tmp(inputs->cols);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for softmax");
    return;
  }

  for (size_t i = 0; i < inputs->rows; i++) {
    float sum = 0;

    float dot[inputs->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = tensor_get(inputs, i, j);
      sum += expf(input);
    }

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = tensor_get(inputs, i, j);
      dot[j] = expf(input) / sum;
    }
    tensor_append_tmp(tmp, dot);
  }

  memcpy(inputs->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  inputs->rows = tmp->rows;
  inputs->cols = tmp->cols;
  inputs->capacity = tmp->capacity;

  tensor_free_tmp(tmp);
}

void tanhh(Tensor *inputs) {
  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return;
  }

  Tensor *tmp = tensor_new_tmp(inputs->cols);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for tanh");
    return;
  }

  for (size_t i = 0; i < inputs->rows; i++) {
    float dot[inputs->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = tensor_get(inputs, i, j);
      dot[j] = tanhf(input);
    }
    tensor_append_tmp(tmp, dot);
  }

  memcpy(inputs->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  inputs->rows = tmp->rows;
  inputs->cols = tmp->cols;
  inputs->capacity = tmp->capacity;

  tensor_free_tmp(tmp);
}

float cross_entropy(Tensor *logits, Tensor *labels) {
  if (!logits || !labels) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !logits ? "logits tensor is NULL"
                             : "labels tensor is NULL");
    return NAN;
  }

  if (!logits->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "logits tensor data is NULL");
    return NAN;
  }

  if (!labels->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "labels tensor data is NULL");
    return NAN;
  }

  if (labels->rows != logits->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "label count (%zu) doesn't match sample count (%zu)",
                         labels->rows, logits->rows);
    return NAN;
  }

  float total_loss = 0.0f;
  for (size_t i = 0; i < logits->rows; i++) {

    float max = tensor_get(logits, i, 0);
    for (size_t j = 0; j < logits->cols; j++) {
      float val = tensor_get(logits, i, j);
      if (val > max)
        max = val;
    }

    float sum = 0.0f;
    for (size_t j = 0; j < logits->cols; j++) {
      float val = tensor_get(logits, i, j);
      sum += expf(val - max);
    }

    float log_sum_exp = logf(sum) + max;

    for (size_t j = 0; j < labels->cols; j++) {
      size_t true_class = (size_t)tensor_get(labels, i, j);

      if (true_class >= logits->cols) {
        ctorch_set_error_fmt(
            CTORCH_ERROR_OUT_OF_BOUNDS,
            "label value %zu at index %zu is out of bounds (must be < %zu)",
            true_class, i, logits->cols);
        return NAN;
      }

      float y = tensor_get(logits, i, true_class);
      float loss = log_sum_exp - y;

      total_loss += loss;
    }
  }

  return total_loss / logits->rows;
}
