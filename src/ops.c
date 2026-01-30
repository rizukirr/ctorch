#include "ops.h"
#include "errors.h"
#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

Tensor *linear(TensorContext *ctx, Tensor *input, Tensor *weight, float *bias) {
  if (!ctx) {
    return NULL;
  }

  if (!input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return NULL;
  }

  if (!weight) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "weight tensor is NULL");
    return NULL;
  }

  if (!bias) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "bias vector is NULL");
    return NULL;
  }

  if (input->cols != weight->rows) {
    if (input->cols == weight->cols) {
      tensor_transpose(weight);
    } else {
      ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                           "dimension mismatch (input: %zux%zu, weight: "
                           "%zux%zu) - expected input.cols == weight.rows",
                           input->rows, input->cols, weight->rows,
                           weight->cols);
      return NULL;
    }
  }

  Tensor *output = tensor_create(ctx, weight->cols);
  if (!output) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate output tensor for linear");
    return NULL;
  }

  for (size_t i = 0; i < input->rows; i++) {
    float row[weight->cols];

    for (size_t k = 0; k < weight->cols; k++) {
      float sum = bias[k];
      for (size_t j = 0; j < input->cols; j++) {
        sum += input->data[i * input->cols + j] *
               weight->data[j * weight->cols + k];
      }
      row[k] = sum;
    }
    tensor_append(ctx, output, row);
  }
  return output;
}

void relu(Tensor *input) {
  if (!input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return;
  }

  if (!input->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return;
  }

  size_t total_elements = input->rows * input->cols;
  for (size_t i = 0; i < total_elements; i++) {
    if (input->data[i] < 0.0f) {
      input->data[i] = 0.0f;
    }
  }
}

void sigmoid(Tensor *input) {
  if (!input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return;
  }

  if (!input->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return;
  }

  size_t total_elements = input->rows * input->cols;
  for (size_t i = 0; i < total_elements; i++) {
    input->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
  }
}

void softmax(Tensor *input) {
  if (!input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return;
  }

  if (!input->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return;
  }

  for (size_t i = 0; i < input->rows; i++) {
    float *row = &input->data[i * input->cols];

    float max_val = row[0];
    for (size_t j = 1; j < input->cols; j++) {
      if (row[j] > max_val) {
        max_val = row[j];
      }
    }

    float sum = 0.0f;
    for (size_t j = 0; j < input->cols; j++) {
      row[j] = expf(row[j] - max_val);
      sum += row[j];
    }

    for (size_t j = 0; j < input->cols; j++) {
      row[j] /= sum;
    }
  }
}

Tensor *softmax_dup(TensorContext *ctx, Tensor *input) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return NULL;
  }

  if (!input->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return NULL;
  }

  Tensor *output = tensor_dup(ctx, input);
  if (!output) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate output tensor for softmax");
    return NULL;
  }
  softmax(output);
  return output;
}

void tanh_(Tensor *input) {
  if (!input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return;
  }

  if (!input->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return;
  }

  size_t total_elements = input->rows * input->cols;
  for (size_t i = 0; i < total_elements; i++) {
    input->data[i] = tanhf(input->data[i]);
  }
}

Tensor *mse_loss(TensorContext *ctx, Tensor *prediction, Tensor *target) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!prediction || !target) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !prediction ? "prediction tensor is NULL"
                                 : "target tensor is NULL");
    return NULL;
  }

  if (!prediction->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "prediction tensor data is NULL");
    return NULL;
  }

  if (!target->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "target tensor data is NULL");
    return NULL;
  }

  if (target->rows != prediction->rows) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_LABEL_MISMATCH,
        "target count (%zu) doesn't match prediction count (%zu)", target->rows,
        prediction->rows);
    return NULL;
  }

  Tensor *loss = tensor_create(ctx, prediction->cols);
  if (!loss) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate output tensor for mse_loss");
    return NULL;
  }

  for (size_t i = 0; i < prediction->rows; i++) {
    float row[prediction->cols];

    for (size_t j = 0; j < prediction->cols; j++) {
      float pred = prediction->data[i * prediction->cols + j];
      float tgt = target->data[i * target->cols + j];
      float diff = pred - tgt;
      row[j] = 0.5f * diff * diff;
    }
    tensor_append(ctx, loss, row);
  }

  return loss;
}

Tensor *cross_entropy(TensorContext *ctx, Tensor *logits, Tensor *target) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!logits || !target) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !logits ? "logits tensor is NULL"
                             : "target tensor is NULL");
    return NULL;
  }

  if (!logits->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "logits tensor data is NULL");
    return NULL;
  }

  if (!target->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "target tensor data is NULL");
    return NULL;
  }

  if (target->rows != logits->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "target count (%zu) doesn't match sample count (%zu)",
                         target->rows, logits->rows);
    return NULL;
  }

  Tensor *loss = tensor_create(ctx, target->cols);
  if (!loss) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate output tensor for cross_entropy");
    return NULL;
  }

  for (size_t i = 0; i < logits->rows; i++) {
    float row[target->cols];

    float *logit_row = &logits->data[i * logits->cols];

    // Find max for numerical stability
    float max_val = logit_row[0];
    for (size_t j = 1; j < logits->cols; j++) {
      if (logit_row[j] > max_val) {
        max_val = logit_row[j];
      }
    }

    // Compute log-sum-exp
    float sum = 0.0f;
    for (size_t j = 0; j < logits->cols; j++) {
      sum += expf(logit_row[j] - max_val);
    }
    float log_sum_exp = logf(sum) + max_val;

    for (size_t j = 0; j < target->cols; j++) {
      size_t true_class = (size_t)target->data[i * target->cols + j];

      if (true_class >= logits->cols) {
        ctorch_set_error_fmt(
            CTORCH_ERROR_OUT_OF_BOUNDS,
            "target value %zu at index %zu is out of bounds (must be < %zu)",
            true_class, i, logits->cols);
        return NULL;
      }

      float logit_val = logit_row[true_class];
      float loss_val = log_sum_exp - logit_val;
      row[j] = loss_val;
    }
    tensor_append(ctx, loss, row);
  }

  return loss;
}
