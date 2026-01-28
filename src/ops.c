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

  // Process each row independently
  for (size_t i = 0; i < input->rows; i++) {
    float *row = &input->data[i * input->cols];

    // Find max for numerical stability
    float max_val = row[0];
    for (size_t j = 1; j < input->cols; j++) {
      if (row[j] > max_val) {
        max_val = row[j];
      }
    }

    // Compute exp(x - max) and sum in single pass
    float sum = 0.0f;
    for (size_t j = 0; j < input->cols; j++) {
      row[j] = expf(row[j] - max_val);
      sum += row[j];
    }

    // Normalize by sum
    for (size_t j = 0; j < input->cols; j++) {
      row[j] /= sum;
    }
  }
}

Tensor *softmax_2dup(TensorContext *ctx, Tensor *input) {
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

Tensor *cross_entropy_backward(TensorContext *ctx, Tensor *output,
                               Tensor *target) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!output || !target) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !output ? "output tensor is NULL"
                             : "target tensor is NULL");
    return NULL;
  }

  if (!output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "output tensor data is NULL");
    return NULL;
  }

  if (!target->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "target tensor data is NULL");
    return NULL;
  }

  if (target->rows != output->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "target count (%zu) doesn't match output count (%zu)",
                         target->rows, output->rows);
    return NULL;
  }

  // Detect target format: class indices (cols=1) vs one-hot (cols=num_classes)
  int is_class_indices = (target->cols == 1);
  int is_one_hot = (target->cols == output->cols);

  if (!is_class_indices && !is_one_hot) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_DIMENSION_MISMATCH,
        "target cols (%zu) must be 1 (class indices) or %zu (one-hot)",
        target->cols, output->cols);
    return NULL;
  }

  Tensor *grad = tensor_create(ctx, output->cols);
  if (!grad) {
    ctorch_set_error(
        CTORCH_ERROR_OUT_OF_MEMORY,
        "failed to allocate gradient tensor for cross_entropy_backward");
    return NULL;
  }

  float batch_size = (float)output->rows;

  for (size_t i = 0; i < output->rows; i++) {
    float row[output->cols];

    if (is_class_indices) {
      // Target is class index: grad = softmax_output - one_hot(target)
      size_t class_idx = (size_t)target->data[i * target->cols];

      for (size_t k = 0; k < output->cols; k++) {
        float prob = output->data[i * output->cols + k];
        row[k] = prob / batch_size;
      }
      row[class_idx] -= 1.0f / batch_size;
    } else {
      // Target is one-hot: grad = softmax_output - target
      for (size_t k = 0; k < output->cols; k++) {
        float prob = output->data[i * output->cols + k];
        float tgt = target->data[i * target->cols + k];
        row[k] = (prob - tgt) / batch_size;
      }
    }

    tensor_append(ctx, grad, row);
  }

  return grad;
}

Tensor *mse_loss_backward(TensorContext *ctx, Tensor *prediction,
                          Tensor *target) {
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

  int is_single_value = (target->cols == 1);
  int is_multi_value = (target->cols == prediction->cols);

  if (!is_single_value && !is_multi_value) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_DIMENSION_MISMATCH,
        "target cols (%zu) must be 1 (single value) or %zu (multi-value)",
        target->cols, prediction->cols);
    return NULL;
  }

  Tensor *grad = tensor_create(ctx, prediction->cols);
  if (!grad) {
    ctorch_set_error(
        CTORCH_ERROR_OUT_OF_MEMORY,
        "failed to allocate gradient tensor for mse_loss_backward");
    return NULL;
  }

  float batch_size = (float)prediction->rows;

  for (size_t i = 0; i < prediction->rows; i++) {
    float row[prediction->cols];

    if (is_single_value) {
      // Target is single value: broadcast to all prediction columns
      float tgt = target->data[i * target->cols];
      for (size_t k = 0; k < prediction->cols; k++) {
        float pred = prediction->data[i * prediction->cols + k];
        row[k] = (pred - tgt) / batch_size;
      }
    } else {
      // Target is multi-value: element-wise gradient
      for (size_t k = 0; k < prediction->cols; k++) {
        float pred = prediction->data[i * prediction->cols + k];
        float tgt = target->data[i * target->cols + k];
        row[k] = (pred - tgt) / batch_size;
      }
    }
    tensor_append(ctx, grad, row);
  }

  return grad;
}

Tensor *relu_backward(TensorContext *ctx, Tensor *grad_output, Tensor *input) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!grad_output || !input) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !grad_output ? "grad_output tensor is NULL"
                                  : "input tensor is NULL");
    return NULL;
  }

  if (!grad_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "grad_output tensor data is NULL");
    return NULL;
  }

  if (!input->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return NULL;
  }

  Tensor *grad = tensor_create(ctx, input->cols);
  if (!grad) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate gradient tensor for relu_backward");
    return NULL;
  }

  for (size_t i = 0; i < input->rows; i++) {
    float row[input->cols];

    for (size_t k = 0; k < input->cols; k++) {
      float val = input->data[i * input->cols + k];
      float upstream = grad_output->data[i * grad_output->cols + k];
      row[k] = val > 0 ? upstream : 0;
    }
    tensor_append(ctx, grad, row);
  }

  return grad;
}

Tensor *sigmoid_backward(TensorContext *ctx, Tensor *grad_output,
                         Tensor *output) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!grad_output || !output) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !grad_output ? "grad_output tensor is NULL"
                                  : "output tensor is NULL");
    return NULL;
  }

  if (!grad_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "grad_output tensor data is NULL");
    return NULL;
  }

  if (!output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "output tensor data is NULL");
    return NULL;
  }

  Tensor *grad = tensor_create(ctx, output->cols);
  if (!grad) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate gradient tensor for sigmoid_backward");
    return NULL;
  }

  for (size_t i = 0; i < output->rows; i++) {
    float row[output->cols];

    for (size_t k = 0; k < output->cols; k++) {
      float sigmoid_val = output->data[i * output->cols + k];
      float upstream = grad_output->data[i * grad_output->cols + k];
      row[k] = sigmoid_val * upstream * (1.0f - sigmoid_val);
    }

    tensor_append(ctx, grad, row);
  }

  return grad;
}

Tensor *tanh_backward(TensorContext *ctx, Tensor *grad_output, Tensor *output) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!grad_output || !output) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !grad_output ? "grad_output tensor is NULL"
                                  : "output tensor is NULL");
    return NULL;
  }

  if (!grad_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "grad_output tensor data is NULL");
    return NULL;
  }

  if (!output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "output tensor data is NULL");
    return NULL;
  }

  Tensor *grad = tensor_create(ctx, output->cols);
  if (!grad) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate gradient tensor for tanh_backward");
    return NULL;
  }

  for (size_t i = 0; i < output->rows; i++) {
    float row[output->cols];

    for (size_t k = 0; k < output->cols; k++) {
      float upstream = grad_output->data[i * grad_output->cols + k];
      float tanh_val = output->data[i * output->cols + k];
      row[k] = upstream * (1.0f - tanh_val * tanh_val);
    }

    tensor_append(ctx, grad, row);
  }

  return grad;
}

Tensor *weight_gradient(TensorContext *ctx, Tensor *input,
                        Tensor *grad_output) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!input || !grad_output) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !input ? "input tensor is NULL"
                            : "grad_output tensor is NULL");
    return NULL;
  }

  if (!input->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "input tensor data is NULL");
    return NULL;
  }

  if (!grad_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "grad_output tensor data is NULL");
    return NULL;
  }

  if (input->rows != grad_output->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "batch size mismatch (input: %zu rows, grad_output: "
                         "%zu rows)",
                         input->rows, grad_output->rows);
    return NULL;
  }

  size_t in_features = input->cols;
  size_t out_features = grad_output->cols;
  size_t batch_size = input->rows;

  Tensor *grad_weight = tensor_create(ctx, out_features);
  if (!grad_weight) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate gradient tensor for weight_gradient");
    return NULL;
  }

  for (size_t j = 0; j < in_features; j++) {
    float row[out_features];

    for (size_t k = 0; k < out_features; k++) {
      float sum = 0.0f;
      for (size_t i = 0; i < batch_size; i++) {
        float x = input->data[i * input->cols + j];
        float grad = grad_output->data[i * grad_output->cols + k];
        sum += x * grad;
      }
      row[k] = sum;
    }
    tensor_append(ctx, grad_weight, row);
  }

  return grad_weight;
}

Tensor *bias_gradient(TensorContext *ctx, Tensor *grad_output) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!grad_output) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "grad_output tensor is NULL");
    return NULL;
  }

  if (!grad_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "grad_output tensor data is NULL");
    return NULL;
  }

  Tensor *grad_bias = tensor_create(ctx, grad_output->cols);
  if (!grad_bias) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate gradient tensor for bias_gradient");
    return NULL;
  }

  // Sum along axis 0 (batch dimension) to get (1, out_features) tensor
  float row[grad_output->cols];

  for (size_t k = 0; k < grad_output->cols; k++) {
    float sum = 0.0f;
    for (size_t i = 0; i < grad_output->rows; i++) {
      sum += grad_output->data[i * grad_output->cols + k];
    }
    row[k] = sum;
  }
  tensor_append(ctx, grad_bias, row);

  return grad_bias;
}

Tensor *input_gradient(TensorContext *ctx, Tensor *grad_output,
                       Tensor *weight) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!grad_output || !weight) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !grad_output ? "grad_output tensor is NULL"
                                  : "weight tensor is NULL");
    return NULL;
  }

  if (!grad_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "grad_output tensor data is NULL");
    return NULL;
  }

  if (!weight->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "weight tensor data is NULL");
    return NULL;
  }

  size_t batch_size = grad_output->rows;
  size_t out_features = grad_output->cols;
  size_t in_features = weight->rows;

  // Check: grad_output->cols (out_features) must match weight->cols
  // (out_features)
  if (out_features != weight->cols) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_DIMENSION_MISMATCH,
        "dimension mismatch (grad_output: %zu cols, weight: %zu cols)",
        out_features, weight->cols);
    return NULL;
  }

  Tensor *grad_input = tensor_create(ctx, in_features);
  if (!grad_input) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate gradient tensor for input_gradient");
    return NULL;
  }

  for (size_t i = 0; i < batch_size; i++) {
    float row[in_features];

    for (size_t j = 0; j < in_features; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < out_features; k++) {
        sum += grad_output->data[i * grad_output->cols + k] *
               weight->data[j * weight->cols + k];
      }
      row[j] = sum;
    }
    tensor_append(ctx, grad_input, row);
  }

  return grad_input;
}
