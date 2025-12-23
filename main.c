#include "ctorch.h"
#include "randn.h"
#include "vector.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define vecpr(v) (vector_print((v), 10, true))

Vector *spiral_data(VectorContext *ctx, size_t n, size_t classes) {
  Vector *X = vector_new(ctx, 3);
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

      vector_append(ctx, X, data);
    }
  }

  return X;
}

int main(void) {
  srand(time(NULL));

  VectorContext *ctx = vector_create();
  Vector *inputs = spiral_data(ctx, 100, 3);
  Vector *weights = vector_randn(ctx, 4, 2);
  float bias[4];
  for (size_t i = 0; i < len(bias); i++) {
    bias[i] = randn();
  }

  if (!inputs) {
    vector_free(ctx);
    return 1;
  }

  Vector *y = vector_get_column(ctx, inputs, 2);
  Vector *x = vector_remove_column(ctx, inputs, 2);

  Vector *logits = affine_transform(ctx, x, weights, bias);
  float loss = cross_entropy_lg(logits, y);

  printf("loss: %f\n", loss);
  printf("\n");

  if (activation_softmax(logits) < 0) {
    vector_free(ctx);
    return 1;
  }

  float ls = cross_entropy(logits, y);
  printf("loss: %f\n", ls);
  printf("\n");

  vecpr(logits);

  vector_free(ctx);
  return 0;
}
