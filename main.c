#include "errors.h"
#include "keras.h"
#include "ops.h"
#include "randn.h"
#include "tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define vecpr(v) (tensor_print((v), 10, true))

Tensor *spiral_data(TensorContext *ctx, size_t n, size_t classes) {
  Tensor *X = tensor_new(ctx, 3);
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

int main(void) {
  srand(time(NULL));

  TensorContext *ctx = tensor_create();
  Tensor *inputs = spiral_data(ctx, 100, 3);

  if (!inputs) {
    tensor_free(ctx);
    return 1;
  }

  Tensor *y = tensor_select(ctx, inputs, 2, AxisColum);
  Tensor *x = tensor_drop(ctx, inputs, 2, AxisColum);

  DenseContext *dense_ctx = dense_init(2);

  Dense *l1 = dense_create(dense_ctx, 3);
  Dense *l2 = dense_create(dense_ctx, 4);
  Dense *l3 = dense_create(dense_ctx, 4);
  Dense *l4 = dense_create(dense_ctx, 4);
  Dense *l5 = dense_create(dense_ctx, 3);

  Tensor *yh1 = dense_forward(dense_ctx, l1, x, ReLU);
  Tensor *yh2 = dense_forward(dense_ctx, l2, yh1, ReLU);
  Tensor *yh3 = dense_forward(dense_ctx, l3, yh2, ReLU);
  Tensor *yh4 = dense_forward(dense_ctx, l4, yh3, ReLU);
  Tensor *yh5 = dense_forward(dense_ctx, l5, yh4, ReLU);

  float loss = cross_entropy(yh5, y);
  printf("Cross entropy loss: %f\n", loss);
  if (isnan(loss)) {
    char *error_msg = ctorch_get_error();
    if (error_msg) {
      printf("Error: %s\n", error_msg);
    }
    dense_free(dense_ctx);
    tensor_free(ctx);
    return 1;
  }

  vecpr(yh5);

  dense_free(dense_ctx);
  tensor_free(ctx);
  return 0;
}
