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
      double r = (double)i / (double)(n - 1);

      /* t = linspace(0, 2.5*2π, N) + (2π * j / classes) + noise */
      double angle_per_class = (2.0 * M_PI) / (double)classes;
      double t = (2.5 * 2.0 * M_PI * i) / (double)(n - 1) +
                 angle_per_class * j + randn() * 0.2;

      double xs = r * sin(t);
      double xc = r * cos(t);
      double class = (double)j;
      double data[3] = {xs, xc, class};

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
  double bias[4];
  for (size_t i = 0; i < len(bias); i++) {
    bias[i] = randn();
  }

  if (!inputs) {
    vector_free(ctx);
    return 1;
  }

  // Vector *y = vector_get_column(ctx, inputs, 2);
  Vector *x = vector_remove_column(ctx, inputs, 2);

  Vector *logits = affine_transform(ctx, x, weights, bias);
  if (activation_softmax(logits) < 0) {
    vector_free(ctx);
    return 1;
  }

  vecpr(logits);

  vector_free(ctx);
  return 0;
}
