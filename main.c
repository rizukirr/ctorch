#include "keras.h"
#include "ops.h"
#include "tensor.h"
#include <math.h>
#include <randn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Generates spiral dataset - hard, interleaved spirals */
Tensor *spiral_data(TensorContext *ctx, size_t n, size_t classes) {
  Tensor *X = tensor_create(ctx, 3);
  if (!X)
    return NULL;

  for (size_t j = 0; j < classes; j++) {
    for (size_t i = 0; i < n; i++) {
      /* r = linspace(0, 1, N) */
      float r = (float)i / (float)(n - 1);

      /* t = linspace(0, 2.5*2π, N) + (2π * j / classes) + noise */
      float angle_per_class = (2.0f * M_PI) / (float)classes;
      float t = (2.5f * 2.0f * M_PI * i) / (float)(n - 1) +
                angle_per_class * j + randn() * 0.2f;

      float xs = r * sinf(t);
      float xc = r * cosf(t);
      float class = (float)j;
      float data[3] = {xs, xc, class};

      tensor_append(ctx, X, data);
    }
  }

  return X;
}

/* Generates blob dataset - easy, linearly separable Gaussian clusters */
Tensor *blob_data(TensorContext *ctx, size_t n, size_t classes) {
  Tensor *X = tensor_create(ctx, 3);
  if (!X)
    return NULL;

  /* Place clusters in a circle around origin */
  for (size_t j = 0; j < classes; j++) {
    float angle = (2.0f * M_PI * j) / (float)classes;
    float center_x = 2.0f * cosf(angle);
    float center_y = 2.0f * sinf(angle);

    for (size_t i = 0; i < n; i++) {
      /* Generate point from Gaussian around cluster center */
      float xs = center_x + randn() * 0.4f;
      float xc = center_y + randn() * 0.4f;
      float class = (float)j;
      float data[3] = {xs, xc, class};

      tensor_append(ctx, X, data);
    }
  }

  return X;
}

/* Generates circles dataset - medium, concentric rings */
Tensor *circles_data(TensorContext *ctx, size_t n, size_t classes) {
  Tensor *X = tensor_create(ctx, 3);
  if (!X)
    return NULL;

  for (size_t j = 0; j < classes; j++) {
    /* Each class is a ring at different radius */
    float radius = 0.5f + (float)j * 1.0f;

    for (size_t i = 0; i < n; i++) {
      /* Random angle */
      float angle = 2.0f * M_PI * ((float)rand() / (float)RAND_MAX);

      /* Point on circle + noise */
      float r = radius + randn() * 0.15f;
      float xs = r * cosf(angle);
      float xc = r * sinf(angle);
      float class = (float)j;
      float data[3] = {xs, xc, class};

      tensor_append(ctx, X, data);
    }
  }

  return X;
}

/* Generates XOR-style dataset - medium, 4 quadrants */
Tensor *xor_data(TensorContext *ctx, size_t n, size_t classes) {
  Tensor *X = tensor_create(ctx, 3);
  if (!X)
    return NULL;

  /* For 2 classes: XOR pattern (top-right + bottom-left vs others) */
  /* For 3+ classes: each class gets a quadrant/region */
  for (size_t j = 0; j < classes; j++) {
    for (size_t i = 0; i < n; i++) {
      float xs, xc;

      if (classes == 2) {
        /* XOR pattern: class 0 in quadrants 1&3, class 1 in quadrants 2&4 */
        int quadrant = (j == 0) ? (i % 2 == 0 ? 0 : 2) : (i % 2 == 0 ? 1 : 3);
        float offset_x = (quadrant == 1 || quadrant == 2) ? 1.0f : -1.0f;
        float offset_y = (quadrant == 0 || quadrant == 1) ? 1.0f : -1.0f;
        xs = offset_x + randn() * 0.3f;
        xc = offset_y + randn() * 0.3f;
      } else {
        /* Each class in a different quadrant/region */
        float angle = (2.0f * M_PI * j) / (float)classes;
        float distance = 1.0f + (float)rand() / (float)RAND_MAX;
        xs = distance * cosf(angle) + randn() * 0.2f;
        xc = distance * sinf(angle) + randn() * 0.2f;
      }

      float class = (float)j;
      float data[3] = {xs, xc, class};
      tensor_append(ctx, X, data);
    }
  }

  return X;
}

int main(void) {
  srand(42); /* Fixed seed for reproducibility */

  const char *datasets[] = {"blob", "circles", "xor", "spiral"};

  for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
    printf("\n=== Testing %s dataset ===\n", datasets[dataset_idx]);

    TensorContext *tensor_ctx = tensor_init();
    if (!tensor_ctx)
      return 1;

    /* Choose dataset */
    Tensor *data;
    switch (dataset_idx) {
    case 0:
      data = blob_data(tensor_ctx, 100, 3);
      break;
    case 1:
      data = circles_data(tensor_ctx, 100, 3);
      break;
    case 2:
      data = xor_data(tensor_ctx, 100, 3);
      break;
    case 3:
      data = spiral_data(tensor_ctx, 100, 3);
      break;
    default:
      return 1;
    }

    Tensor *x = tensor_drop(tensor_ctx, data, 2, AxisColumn);
    Tensor *y = tensor_select(tensor_ctx, data, 2, AxisColumn);

    DenseContext *dense_ctx = dense_init(x->cols);
    if (!dense_ctx)
      return 1;

    dense_create(dense_ctx, 4, ReLU);
    dense_create(dense_ctx, 4, ReLU);
    dense_create(dense_ctx, 4, ReLU);
    dense_create(dense_ctx, 3, Linear);

    for (size_t i = 0; i < 1000; i++) {
      Tensor *output = dense_forward(dense_ctx, x);
      Tensor *softmax_output = softmax_dup(tensor_ctx, output);

      Tensor *loss = cross_entropy(tensor_ctx, output, y);
      float loss_avg = tensor_avg(loss);

      Tensor *grad_output =
          cross_entropy_backward(tensor_ctx, softmax_output, y);
      dense_backward(dense_ctx, grad_output, 0.01f, Adam);

      if (i % 200 == 0 || i == 999) {
        Tensor *preds = predict(dense_ctx, x);
        float acc = accuracy(dense_ctx, preds, y);
        printf("Iter %4zu: loss=%.6f, accuracy=%.2f%%\n", i, loss_avg,
               acc * 100);
      }
    }

    dense_free(dense_ctx);
    tensor_free(tensor_ctx);
  }

  return 0;
}
