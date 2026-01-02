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

  // Create array of layers for optimizer
  Dense *layers[] = {l1, l2, l3, l4, l5};
  size_t num_layers = 5;

  // Training hyperparameters
  size_t num_epochs = 100;
  float learning_rate = 0.1f;

  printf("Training neural network on spiral dataset...\n");
  printf("Epochs: %zu, Learning rate: %.3f\n\n", num_epochs, learning_rate);

  // Training loop
  for (size_t epoch = 0; epoch < num_epochs; epoch++) {
    // Zero gradients
    for (size_t i = 0; i < num_layers; i++) {
      dense_zero_grad(layers[i]);
    }

    // Forward pass
    Tensor *yh1 = dense_forward(dense_ctx, l1, x, ReLU);
    Tensor *yh2 = dense_forward(dense_ctx, l2, yh1, ReLU);
    Tensor *yh3 = dense_forward(dense_ctx, l3, yh2, ReLU);
    Tensor *yh4 = dense_forward(dense_ctx, l4, yh3, ReLU);
    Tensor *yh5 =
        dense_forward(dense_ctx, l5, yh4, None); // No activation for logits

    // Compute loss
    Tensor *losses = cross_entropy(ctx, yh5, y);
    if (!losses) {
      char *error_msg = ctorch_get_error();
      if (error_msg) {
        fprintf(stderr, "Error: %s\n", error_msg);
      }
      dense_free(dense_ctx);
      tensor_free(ctx);
      return 1;
    }

    // Calculate average loss
    float avg_loss = tensor_avg(losses);

    // Calculate accuracy
    float accuracy = tensor_accuracy(yh5, y);

    // Print progress every 10 epochs
    if (epoch % 10 == 0 || epoch == num_epochs - 1) {
      printf("Epoch %3zu: Loss = %.4f, Accuracy = %.2f%%\n", epoch, avg_loss,
             accuracy);
    }

    // Backward pass
    Tensor *grad_yh5 = cross_entropy_backward(ctx, yh5, y);
    if (!grad_yh5) {
      char *error_msg = ctorch_get_error();
      if (error_msg) {
        fprintf(stderr, "Backward error: %s\n", error_msg);
      }
      dense_free(dense_ctx);
      tensor_free(ctx);
      return 1;
    }

    Tensor *grad_yh4 = dense_backward(dense_ctx, l5, grad_yh5);
    Tensor *grad_yh3 = dense_backward(dense_ctx, l4, grad_yh4);
    Tensor *grad_yh2 = dense_backward(dense_ctx, l3, grad_yh3);
    Tensor *grad_yh1 = dense_backward(dense_ctx, l2, grad_yh2);
    Tensor *grad_x = dense_backward(dense_ctx, l1, grad_yh1);
    (void)grad_x; // Unused, suppress warning

    // Optimizer step
    int ret = sgd_step(dense_ctx, layers, num_layers, learning_rate);
    if (ret < 0) {
      char *error_msg = ctorch_get_error();
      if (error_msg) {
        fprintf(stderr, "Optimizer error: %s\n", error_msg);
      }
      dense_free(dense_ctx);
      tensor_free(ctx);
      return 1;
    }
  }

  printf("\nTraining complete!\n");

  dense_free(dense_ctx);
  tensor_free(ctx);
  return 0;
}
