#include "ops.h"
#include "arena.h"
#include "errors.h"
#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// Access to TensorContext internals for gradient allocation
struct TensorContext {
  bool requires_grad;
  Arena *arena;
};

float tensor_avg(Tensor *src) {
  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return 0.0f;
  }

  float avg = 0.0f;
  for (size_t i = 0; i < src->rows; i++) {
    for (size_t j = 0; j < src->cols; j++) {
      avg += tensor_get(src, i, j);
    }
  }
  avg /= (float)(src->rows * src->cols);
  return avg;
}

float tensor_accuracy(Tensor *y_pred, Tensor *y_true) {
  if (!y_pred || !y_true) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !y_pred ? "y_pred tensor is NULL"
                             : "y_true tensor is NULL");
    return 0.0f;
  }

  if (y_pred->rows != y_true->rows) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_DIMENSION_MISMATCH,
        "dimension mismatch (y_pred: %zux%zu, y_true: %zux%zu) - "
        "expected y_pred.cols == y_true.rows",
        y_pred->rows, y_pred->cols, y_true->rows, y_true->cols);
    return 0.0f;
  }

  size_t correct = 0;
  for (size_t i = 0; i < y_pred->rows; i++) {
    // Find predicted class (argmax)
    size_t pred_class = 0;
    float max_prob = tensor_get(y_pred, i, 0);

    for (size_t j = 1; j < y_pred->cols; j++) {
      float prob = tensor_get(y_pred, i, j);
      if (prob > max_prob) {
        max_prob = prob;
        pred_class = j;
      }
    }

    // Compare with true class
    for (size_t j = 0; j < y_true->cols; j++) {
      size_t true_class = (size_t)tensor_get(y_true, i, j);
      if (pred_class == true_class) {
        correct++;
      }
    }
  }
  float ret = (float)correct / (float)y_pred->rows * 100.0f;
  return ret;
}

Tensor *tensor_mul(TensorContext *ctx, Tensor *a, Tensor *b) {
  if (!ctx) {
    return NULL;
  }

  if (!a) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor a is NULL");
    return NULL;
  }

  if (!b) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor b is NULL");
    return NULL;
  }

  if (a->cols != b->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "dimension mismatch (a: %zux%zu, b: %zux%zu) - "
                         "expected a.cols == b.rows",
                         a->rows, a->cols, b->rows, b->cols);
    return NULL;
  }

  Tensor *outputs = tensor_new(ctx, b->cols);
  if (!outputs)
    return NULL;

  for (size_t i = 0; i < a->rows; i++) {
    float dot[b->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t k = 0; k < b->cols; k++) {
      for (size_t j = 0; j < a->cols; j++) {
        float input = tensor_get(a, i, j);
        float weight = tensor_get(b, j, k);
        dot[k] += input * weight;
      }
    }
    tensor_append(ctx, outputs, dot);
  }
  return outputs;
}

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

int relu(Tensor *inputs) {
  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return -1;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return -1;
  }

  Tensor *tmp = tensor_new_tmp(inputs->cols);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for ReLU");
    return -1;
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
  return 0;
}

int sigmoid(Tensor *inputs) {
  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return -1;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return -1;
  }

  Tensor *tmp = tensor_new_tmp(inputs->cols);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for sigmoid");
    return -1;
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
  return 0;
}

int softmax(Tensor *inputs) {
  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return CTORCH_ERROR_NULL_DATA;
  }

  Tensor *tmp = tensor_new_tmp(inputs->cols);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for softmax");
    return CTORCH_ERROR_OUT_OF_MEMORY;
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
  return 0;
}

int tanhh(Tensor *inputs) {
  if (!inputs) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "input tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return CTORCH_ERROR_NULL_DATA;
  }

  Tensor *tmp = tensor_new_tmp(inputs->cols);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for tanh");
    return CTORCH_ERROR_OUT_OF_MEMORY;
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
  return 0;
}

Tensor *cross_entropy(TensorContext *ctx, Tensor *logits, Tensor *labels) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!logits || !labels) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !logits ? "logits tensor is NULL"
                             : "labels tensor is NULL");
    return NULL;
  }

  if (!logits->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "logits tensor data is NULL");
    return NULL;
  }

  if (!labels->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "labels tensor data is NULL");
    return NULL;
  }

  if (labels->rows != logits->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "label count (%zu) doesn't match sample count (%zu)",
                         labels->rows, logits->rows);
    return NULL;
  }

  Tensor *losses = tensor_new(ctx, 1);
  if (!losses) {
    return NULL;
  }

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
        return NULL;
      }

      float y = tensor_get(logits, i, true_class);
      float loss = log_sum_exp - y;

      float loss_data[1] = {loss};
      tensor_append(ctx, losses, loss_data);
    }
  }

  return losses;
}

Tensor *tensor_subtract(TensorContext *ctx, Tensor *a, Tensor *b) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!a || !b) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !a ? "tensor a is NULL" : "tensor b is NULL");
    return NULL;
  }

  if (a->rows != b->rows || a->cols != b->cols) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "dimension mismatch (a: %zux%zu, b: %zux%zu) - "
                         "expected same shape",
                         a->rows, a->cols, b->rows, b->cols);
    return NULL;
  }

  Tensor *v = tensor_new(ctx, a->cols);
  if (!v)
    return NULL;

  for (size_t i = 0; i < a->rows; i++) {
    float row_data[a->cols];
    for (size_t j = 0; j < a->cols; j++) {
      row_data[j] = tensor_get(a, i, j) - tensor_get(b, i, j);
    }
    tensor_append(ctx, v, row_data);
  }

  return v;
}

// ========== BACKWARD PROPAGATION OPERATIONS ==========

int relu_backward(TensorContext *ctx, Tensor *grad_output,
                  Tensor *pre_activation, Tensor *grad_input) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!grad_output || !pre_activation || !grad_input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     "one or more tensor parameters are NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (grad_output->rows != pre_activation->rows ||
      grad_output->cols != pre_activation->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH,
                     "grad_output and pre_activation must have same shape");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  // grad_input = grad_output * (pre_activation > 0)
  for (size_t i = 0; i < grad_output->rows * grad_output->cols; i++) {
    grad_input->data[i] =
        grad_output->data[i] * (pre_activation->data[i] > 0 ? 1.0f : 0.0f);
  }

  return 0;
}

int sigmoid_backward(TensorContext *ctx, Tensor *grad_output, Tensor *output,
                     Tensor *grad_input) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!grad_output || !output || !grad_input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     "one or more tensor parameters are NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (grad_output->rows != output->rows || grad_output->cols != output->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH,
                     "grad_output and output must have same shape");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  // grad_input = grad_output * output * (1 - output)
  for (size_t i = 0; i < grad_output->rows * grad_output->cols; i++) {
    float y = output->data[i];
    grad_input->data[i] = grad_output->data[i] * y * (1.0f - y);
  }

  return 0;
}

int tanh_backward(TensorContext *ctx, Tensor *grad_output, Tensor *output,
                  Tensor *grad_input) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!grad_output || !output || !grad_input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     "one or more tensor parameters are NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (grad_output->rows != output->rows || grad_output->cols != output->cols) {
    ctorch_set_error(CTORCH_ERROR_DIMENSION_MISMATCH,
                     "grad_output and output must have same shape");
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  // grad_input = grad_output * (1 - output^2)
  for (size_t i = 0; i < grad_output->rows * grad_output->cols; i++) {
    float y = output->data[i];
    grad_input->data[i] = grad_output->data[i] * (1.0f - y * y);
  }

  return 0;
}

Tensor *affine_backward(TensorContext *ctx, Tensor *grad_output, Tensor *inputs,
                        Tensor *weights, Tensor **grad_inputs,
                        Tensor **grad_weights, float **grad_bias) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!grad_output || !inputs || !weights) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     "one or more tensor parameters are NULL");
    return NULL;
  }

  // grad_inputs = grad_output @ weights^T
  // First transpose weights
  Tensor *weights_T = tensor_new_tmp(weights->rows);
  if (!weights_T) {
    ctorch_set_error(
        CTORCH_ERROR_OUT_OF_MEMORY,
        "failed to allocate temporary tensor for weights transpose");
    return NULL;
  }

  for (size_t i = 0; i < weights->cols; i++) {
    float row[weights->rows];
    for (size_t j = 0; j < weights->rows; j++) {
      row[j] = tensor_get(weights, j, i);
    }
    tensor_append_tmp(weights_T, row);
  }

  *grad_inputs = tensor_mul(ctx, grad_output, weights_T);
  tensor_free_tmp(weights_T);

  if (!*grad_inputs)
    return NULL;

  // grad_weights = inputs^T @ grad_output
  Tensor *inputs_T = tensor_new_tmp(inputs->rows);
  if (!inputs_T) {
    ctorch_set_error(
        CTORCH_ERROR_OUT_OF_MEMORY,
        "failed to allocate temporary tensor for inputs transpose");
    return NULL;
  }

  for (size_t i = 0; i < inputs->cols; i++) {
    float row[inputs->rows];
    for (size_t j = 0; j < inputs->rows; j++) {
      row[j] = tensor_get(inputs, j, i);
    }
    tensor_append_tmp(inputs_T, row);
  }

  *grad_weights = tensor_mul(ctx, inputs_T, grad_output);
  tensor_free_tmp(inputs_T);

  if (!*grad_weights)
    return NULL;

  // grad_bias = sum(grad_output, axis=0) - column-wise sum
  *grad_bias = arena_alloc(ctx->arena, grad_output->cols * sizeof(float),
                           ARENA_ALIGNOF(float));
  if (!*grad_bias) {
    ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                     "failed to allocate gradient bias array");
    return NULL;
  }

  // Initialize to zero
  memset(*grad_bias, 0, grad_output->cols * sizeof(float));

  // Sum over rows (axis=0)
  for (size_t col = 0; col < grad_output->cols; col++) {
    float sum = 0.0f;
    for (size_t row = 0; row < grad_output->rows; row++) {
      sum += tensor_get(grad_output, row, col);
    }
    (*grad_bias)[col] = sum;
  }

  return *grad_inputs;
}

Tensor *cross_entropy_backward(TensorContext *ctx, Tensor *logits,
                               Tensor *labels) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!logits || !labels) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !logits ? "logits tensor is NULL"
                             : "labels tensor is NULL");
    return NULL;
  }

  if (labels->rows != logits->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "label count (%zu) doesn't match sample count (%zu)",
                         labels->rows, logits->rows);
    return NULL;
  }

  // Compute softmax(logits) in-place copy
  Tensor *probs = tensor_copy(ctx, logits);
  if (!probs)
    return NULL;

  int ret = softmax(probs);
  if (ret < 0)
    return NULL;

  // grad = (probs - one_hot(labels)) / batch_size
  Tensor *grad = tensor_new(ctx, logits->cols);
  if (!grad)
    return NULL;

  float batch_size = (float)logits->rows;

  for (size_t i = 0; i < logits->rows; i++) {
    float row_grad[logits->cols];

    // Start with softmax probabilities
    for (size_t j = 0; j < logits->cols; j++) {
      row_grad[j] = tensor_get(probs, i, j);
    }

    // Subtract 1 at true class positions
    for (size_t k = 0; k < labels->cols; k++) {
      size_t true_class = (size_t)tensor_get(labels, i, k);
      if (true_class < logits->cols) {
        row_grad[true_class] -= 1.0f;
      }
    }

    // Divide by batch size
    for (size_t j = 0; j < logits->cols; j++) {
      row_grad[j] /= batch_size;
    }

    tensor_append(ctx, grad, row_grad);
  }

  return grad;
}
