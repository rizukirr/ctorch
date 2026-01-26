#include "ops.h"
#include "errors.h"
#include "tensor.h"
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
  if (!outputs){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for affine_transform");
    return NULL;
  }

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
    float dot[inputs->cols];
    memset(dot, 0, sizeof(dot));

    // Find max for numerical stability
    float max = tensor_get(inputs, i, 0);
    for (size_t j = 1; j < inputs->cols; j++) {
      float val = tensor_get(inputs, i, j);
      if (val > max)
        max = val;
    }

    // Compute sum of exp(input - max)
    float sum = 0.0f;
    for (size_t j = 0; j < inputs->cols; j++) {
      float input = tensor_get(inputs, i, j);
      sum += expf(input - max);
    }

    // Compute softmax with numerical stability
    for (size_t j = 0; j < inputs->cols; j++) {
      float input = tensor_get(inputs, i, j);
      dot[j] = expf(input - max) / sum;
    }
    tensor_append_tmp(tmp, dot);
  }

  memcpy(inputs->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  inputs->rows = tmp->rows;
  inputs->cols = tmp->cols;

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

  tensor_free_tmp(tmp);
}

Tensor *squared_error(TensorContext *ctx, Tensor *y_pred, Tensor *y_true) {
  if (!y_pred || !y_true) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !y_pred ? "y_pred tensor is NULL"
                             : "y_true tensor is NULL");
    return NULL;
  }

  if (!y_pred->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "logits tensor data is NULL");
    return NULL;
  }

  if (!y_true->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "y_true tensor data is NULL");
    return NULL;
  }

  if (y_true->rows != y_pred->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "label count (%zu) doesn't match sample count (%zu)",
                         y_true->rows, y_pred->rows);
    return NULL;
  }

  Tensor *loss = tensor_new(ctx, y_pred->cols);
  if (!loss){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for squared_error");
    return NULL;
  }

  for (size_t i = 0; i < y_pred->rows; i++) {
    float temp[y_pred->cols];
    memset(temp, 0, sizeof(temp));

    for (size_t j = 0; j < y_pred->cols; j++) {
      float val = tensor_get(y_pred, i, j);
      float diff = val - tensor_get(y_true, i, j);
      temp[j] = 0.5f * powf(diff, 2);
    }
    tensor_append(ctx, loss, temp);
  }

  return loss;
}

Tensor *cross_entropy(TensorContext *ctx, Tensor *logits, Tensor *y_true) {
  if (!logits || !y_true) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !logits ? "logits tensor is NULL"
                             : "y_true tensor is NULL");
    return NULL;
  }

  if (!logits->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "logits tensor data is NULL");
    return NULL;
  }

  if (!y_true->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "y_true tensor data is NULL");
    return NULL;
  }

  if (y_true->rows != logits->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "label count (%zu) doesn't match sample count (%zu)",
                         y_true->rows, logits->rows);
    return NULL;
  }

  Tensor *loss = tensor_new(ctx, logits->cols);
  if (!loss){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for cross_entropy");
    return NULL;
  }

  for (size_t i = 0; i < logits->rows; i++) {
    float temp[logits->cols];
    memset(temp, 0, sizeof(temp));

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

    for (size_t j = 0; j < y_true->cols; j++) {
      size_t true_class = (size_t)tensor_get(y_true, i, j);

      if (true_class >= logits->cols) {
        ctorch_set_error_fmt(
            CTORCH_ERROR_OUT_OF_BOUNDS,
            "label value %zu at index %zu is out of bounds (must be < %zu)",
            true_class, i, logits->cols);
        return NULL;
      }

      float y = tensor_get(logits, i, true_class);
      float l = log_sum_exp - y;
      temp[j] = l;
    }
    tensor_append(ctx, loss, temp);
  }

  return loss;
}



// Backward pass
Tensor *cross_entropy_backward(TensorContext *ctx, Tensor *softmax_output, Tensor *y_true) {
  if (!softmax_output || !y_true) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !softmax_output ? "y_pred tensor is NULL"
                             : "y_true tensor is NULL");
    return NULL;
  }

  if (!softmax_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "logits tensor data is NULL");
    return NULL;
  }

  if (!y_true->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "y_true tensor data is NULL");
    return NULL;
  }

  if (y_true->rows != softmax_output->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "label count (%zu) doesn't match sample count (%zu)",
                         y_true->rows, softmax_output->rows);
    return NULL;
  }

  Tensor *dL_dZ = tensor_new(ctx, softmax_output->cols);
  if (!dL_dZ){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for cross_entropy backward");
    return NULL;
  }

  float N = (float)softmax_output->rows;

  for(size_t i = 0; i < softmax_output->rows; i++){
    float temp[softmax_output->cols];
    memset(temp, 0, sizeof(temp));

    for(size_t k = 0; k < softmax_output->cols; k++){
      float p = tensor_get(softmax_output, i, k);
      float y = tensor_get(y_true, i, k);
      float grad = (p - y) / N;
      temp[k] = grad;
    }
    tensor_append(ctx, dL_dZ, temp);
  }

  return dL_dZ;
}

Tensor *squared_error_backward(TensorContext *ctx, Tensor *y_pred, Tensor *y_true) {
  if (!y_pred || !y_true) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !y_pred ? "y_pred tensor is NULL"
                             : "y_true tensor is NULL");
    return NULL;
  }

  if (!y_pred->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "logits tensor data is NULL");
    return NULL;
  }

  if (!y_true->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "y_true tensor data is NULL");
    return NULL;
  }

  if (y_true->rows != y_pred->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_LABEL_MISMATCH,
                         "label count (%zu) doesn't match sample count (%zu)",
                         y_true->rows, y_pred->rows);
    return NULL;
  }

  Tensor *dL_dy = tensor_new(ctx, y_pred->cols);
  if (!dL_dy){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for squared_error backward");
    return NULL;
  }

  float N = (float)y_pred->rows;

  for (size_t i = 0; i < y_pred->rows; i++) {
    float temp[y_pred->cols];
    memset(temp, 0, sizeof(temp));

    for(size_t k = 0; k < y_pred->cols; k++){
      float ypred_ik = tensor_get(y_pred, i, k);
      float ytrue_ik = tensor_get(y_true, i, k);
      float grad = (ypred_ik - ytrue_ik) / N;
      temp[k] = grad;
    }
    tensor_append(ctx, dL_dy, temp);
  }

  return dL_dy;
}

Tensor *relu_backward(TensorContext *ctx, Tensor *loss_grad, Tensor *logits){
  if (!loss_grad || !logits) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !loss_grad ? "loss_grad tensor is NULL"
                                : "logits tensor is NULL");
    return NULL;
  }

  if (!loss_grad->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "loss_grad tensor data is NULL");
    return NULL;
  }

  if (!logits->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "logits tensor data is NULL");
    return NULL;
  }

  Tensor *dL_dZ = tensor_new(ctx, logits->cols);
  if (!dL_dZ){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for relu backward");
    return NULL;
  }
  for (size_t i = 0; i < logits->rows; i++) {
    float temp[logits->cols];
    memset(temp, 0, sizeof(temp));

    for (size_t k = 0; k < logits->cols; k++) {
      float val = tensor_get(logits, i, k);
      float grad = tensor_get(loss_grad, i, k);
      temp[k] = val > 0 ? grad : 0;
    }
    tensor_append(ctx, dL_dZ, temp);
  }

  return dL_dZ;
}

Tensor *sigmoid_backward(TensorContext *ctx, Tensor *upstream_grad, Tensor *sigmoid_output) {
  if (!upstream_grad || !sigmoid_output) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !upstream_grad ? "upstream_grad tensor is NULL"
                                    : "sigmoid_output tensor is NULL");
    return NULL;
  }

  if (!upstream_grad->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "upstream_grad tensor data is NULL");
    return NULL;
  }

  if (!sigmoid_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "sigmoid_output tensor data is NULL");
    return NULL;
  }

  Tensor *dL_dZ = tensor_new(ctx, sigmoid_output->cols);
  if (!dL_dZ){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for sigmoid backward");
    return NULL;
  }

  for (size_t i = 0; i < sigmoid_output->rows; i++) {
    float temp[sigmoid_output->cols];
    memset(temp, 0, sizeof(temp));

    for (size_t k = 0; k < sigmoid_output->cols; k++) {
      float val = tensor_get(sigmoid_output, i, k);
      float grad = tensor_get(upstream_grad, i, k);
      temp[k] = val * grad * (1.0f - val);
    }

    tensor_append(ctx, dL_dZ, temp);
  }

  return dL_dZ;
}

Tensor *tanh_backward(TensorContext *ctx, Tensor *upstream_grad, Tensor *tanh_output) {
  if (!upstream_grad || !tanh_output) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !upstream_grad ? "upstream_grad tensor is NULL"
                                    : "tanh_output tensor is NULL");
    return NULL;
  }

  if (!upstream_grad->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "upstream_grad tensor data is NULL");
    return NULL;
  }

  if (!tanh_output->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "tanh_output tensor data is NULL");
    return NULL;
  }

  Tensor *dL_dZ = tensor_new(ctx, tanh_output->cols);
  if (!dL_dZ){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for tanh backward");
    return NULL;
  }

  for (size_t i = 0; i < tanh_output->rows; i++) {
    float temp[tanh_output->cols];
    memset(temp, 0, sizeof(temp));

    for (size_t k = 0; k < tanh_output->cols; k++) {
      float upstream = tensor_get(upstream_grad, i, k);
      float tahn_val = tensor_get(tanh_output, i, k);

      temp[k] = upstream * (1.0f - powf(tahn_val, 2));
    }

    tensor_append(ctx, dL_dZ, temp);
  }

  return dL_dZ;
}

Tensor *weight_gradient(TensorContext *ctx, Tensor *inputs, Tensor *upstream_grad){
  if (!inputs || !upstream_grad) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !inputs ? "inputs tensor is NULL"
                             : "upstream_grad tensor is NULL");
    return NULL;
  }

  if (!inputs->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "inputs tensor data is NULL");
    return NULL;
  }

  if (!upstream_grad->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "upstream_grad tensor data is NULL");
    return NULL;
  }

  if (inputs->rows != upstream_grad->rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "batch size mismatch (inputs: %zu rows, upstream_grad: %zu rows)",
                         inputs->rows, upstream_grad->rows);
    return NULL;
  }

  size_t dim_in = inputs->cols;
  size_t dim_out = upstream_grad->cols;
  size_t N = inputs->rows;

  Tensor *dL_dW = tensor_new(ctx, dim_out);
  if (!dL_dW){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for affine_transform backward");
    return NULL;
  }


  for(size_t j = 0; j < dim_in; j++){
    float temp[dim_out];
    memset(temp, 0, sizeof(temp));

    for(size_t k = 0; k < dim_out; k++){
      float sum = 0.0f;
      for(size_t i = 0; i < N; i++){
        float x = tensor_get(inputs, i, j);
        float grad = tensor_get(upstream_grad, i, k);

        sum += x * grad;
      }
      temp[k] = sum;
    }
    tensor_append(ctx, dL_dW, temp);
  }

  return dL_dW;
}

Tensor *bias_gradient(TensorContext *ctx, Tensor *upstream_grad) {
  if (!upstream_grad) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "upstream_grad tensor is NULL");
    return NULL;
  }

  if (!upstream_grad->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "upstream_grad tensor data is NULL");
    return NULL;
  }

  Tensor *dL_db = tensor_new(ctx, upstream_grad->cols);
  if (!dL_db){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for bias_gradient");
    return NULL;
  }

  // Sum along axis 0 (batch dimension) to get (1, D_out) tensor
  float temp[upstream_grad->cols];
  memset(temp, 0, sizeof(temp));

  for(size_t k = 0; k < upstream_grad->cols; k++){
    float sum = 0.0f;
    for(size_t i = 0; i < upstream_grad->rows; i++){
      sum += tensor_get(upstream_grad, i, k);
    }
    temp[k] = sum;
  }
  tensor_append(ctx, dL_db, temp);

  return dL_db;
}

Tensor *input_gradient(TensorContext *ctx, Tensor *upstream_grad, Tensor *weights){
  if (!upstream_grad || !weights) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                      !upstream_grad ? "upstream_grad tensor is NULL"
                                     : "weights tensor is NULL");
    return NULL;
  }

  if (!upstream_grad->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "upstream_grad tensor data is NULL");
    return NULL;
  }

  if (!weights->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "weights tensor data is NULL");
    return NULL;
  }

  size_t N = upstream_grad->rows;
  size_t dim_out = upstream_grad->cols;
  size_t dim_in = weights->rows;

  // Check: upstream_grad->cols (D_out) must match weights->cols (D_out)
  if (dim_out != weights->cols) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "dimension mismatch (upstream_grad: %zu cols, weights: %zu cols)",
                         dim_out, weights->cols);
    return NULL;
  }

  Tensor *dL_dX = tensor_new(ctx, dim_in);
  if (!dL_dX){
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for input_gradient");
    return NULL;
  }

  for(size_t i = 0; i < N; i++){
    float temp[dim_in];
    memset(temp, 0, sizeof(temp));

    for(size_t j = 0; j < dim_in; j++){
      float sum = 0.0f;
      for(size_t k = 0; k < dim_out; k++){
        sum += tensor_get(upstream_grad, i, k) * tensor_get(weights, j, k);
      }

      temp[j] = sum;
    }
    tensor_append(ctx, dL_dX, temp);
  }

  return dL_dX;
}
