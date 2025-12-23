#include "ctorch.h"
#include "vector.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

Vector *affine_transform(VectorContext *ctx, Vector *inputs, Vector *weights,
                         float *bias) {
  if (inputs->cols != weights->rows) {
    if (inputs->cols == weights->cols) {
      vector_transpose(weights);
    } else {
      fprintf(stderr, "dimension mismatch (%zu, %zu) (%zu, %zu)\n",
              inputs->cols, inputs->rows, weights->cols, weights->rows);
      return NULL;
    }
  }

  Vector *outputs = vector_new(ctx, weights->cols);
  if (!outputs)
    return NULL;

  for (size_t i = 0; i < inputs->rows; i++) {
    float dot[weights->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t k = 0; k < weights->cols; k++) {
      for (size_t j = 0; j < inputs->cols; j++) {
        float input = vector_get(inputs, i, j);
        float weight = vector_get(weights, j, k);
        dot[k] += input * weight;
      }
      dot[k] += bias[k];
    }
    vector_append(ctx, outputs, dot);
  }
  return outputs;
}

int activation_ReLU(Vector *inputs) {
  if (!inputs)
    return -1;

  Vector *tmp = vector_new_tmp(inputs->cols);
  if (!tmp)
    return -1;

  for (size_t i = 0; i < inputs->rows; i++) {
    float dot[inputs->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = vector_get(inputs, i, j);
      dot[j] = input > 0 ? input : 0;
    }
    vector_append_tmp(tmp, dot);
  }

  memcpy(inputs->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  inputs->rows = tmp->rows;
  inputs->cols = tmp->cols;
  inputs->capacity = tmp->capacity;

  vector_free_tmp(tmp);
  return 0;
}

int activation_sigmoid(Vector *inputs) {
  if (!inputs)
    return -1;

  Vector *tmp = vector_new_tmp(inputs->cols);
  if (!tmp)
    return -1;

  for (size_t i = 0; i < inputs->rows; i++) {
    float dot[inputs->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = vector_get(inputs, i, j);
      dot[j] = 1 / (1 + expf(-input));
    }
    vector_append_tmp(tmp, dot);
  }

  memcpy(inputs->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  inputs->rows = tmp->rows;
  inputs->cols = tmp->cols;
  inputs->capacity = tmp->capacity;

  vector_free_tmp(tmp);

  return 0;
}

int activation_softmax(Vector *inputs) {
  if (!inputs)
    return -1;

  Vector *tmp = vector_new_tmp(inputs->cols);
  if (!tmp)
    return -1;

  for (size_t i = 0; i < inputs->rows; i++) {
    float sum = 0;

    float dot[inputs->cols];
    memset(dot, 0, sizeof(dot));

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = vector_get(inputs, i, j);
      sum += expf(input);
    }

    for (size_t j = 0; j < inputs->cols; j++) {
      float input = vector_get(inputs, i, j);
      dot[j] = expf(input) / sum;
    }
    vector_append_tmp(tmp, dot);
  }

  memcpy(inputs->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  inputs->rows = tmp->rows;
  inputs->cols = tmp->cols;
  inputs->capacity = tmp->capacity;

  vector_free_tmp(tmp);

  return 0;
}

float cross_entropy_lg(Vector *logits, Vector *labels) {
  if (!logits || !labels)
    return -1;

  float total_loss = 0.0f;
  for (size_t i = 0; i < logits->rows; i++) {

    float max = vector_get(logits, i, 0);
    for (size_t j = 0; j < logits->cols; j++) {
      float val = vector_get(logits, i, j);
      if (val > max)
        max = val;
    }

    float sum = 0.0f;
    for (size_t j = 0; j < logits->cols; j++) {
      float val = vector_get(logits, i, j);
      sum += expf(val - max);
    }

    float log_sum_exp = logf(sum) + max;

    size_t true_class = (size_t)labels->data[i];
    float y = vector_get(logits, i, true_class);
    total_loss += log_sum_exp - y;
  }

  return total_loss / logits->rows;
}

#define EPSILON 1e-7f

float cross_entropy(Vector *logits, Vector *labels) {
  if (!logits || !labels)
    return -1;

  float loss = 0.0f;

  for (size_t i = 0; i < logits->rows; i++) {
    size_t true_class = (size_t)labels->data[i];
    float val = vector_get(logits, i, true_class);
    if (val < EPSILON)
      val = EPSILON;

    loss -= logf(val);
  }

  return loss / logits->rows;
}
