#include "keras.h"
#include "ops.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
  srand(time(NULL));
  TensorContext *ctx = tensor_create();

  // === SETUP: Simple 2-layer network (2 -> 4 -> 3) ===

  // Input data: 4 samples, 2 features
  Tensor *X = tensor_new(ctx, 2);
  float x_data[4][2] = {{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}, {0.7f, 0.8f}};
  for (int i = 0; i < 4; i++)
    tensor_append(ctx, X, x_data[i]);

  // Labels: one-hot encoded (4 samples, 3 classes)
  Tensor *Y = tensor_new(ctx, 3);
  float y_data[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}};
  for (int i = 0; i < 4; i++)
    tensor_append(ctx, Y, y_data[i]);

  // === KERAS API TEST ===
  printf("\n\n=== Testing Keras API with dense_backward ===\n");

  // Create keras-style network: 2 -> 4 (ReLU) -> 3 (Softmax)
  DenseContext *keras_ctx = dense_init(2);
  if (!keras_ctx) {
    printf("Failed to create keras context\n");
    return 1;
  }

  dense_create(keras_ctx, 4, ReLU);
  dense_create(keras_ctx, 3, Softmax);

  // Create separate tensor context for inputs/labels
  TensorContext *data_ctx = tensor_create();

  // Input data: 4 samples, 2 features
  Tensor *keras_X = tensor_new(data_ctx, 2);
  for (int i = 0; i < 4; i++)
    tensor_append(data_ctx, keras_X, x_data[i]);

  // Labels: one-hot encoded (4 samples, 3 classes)
  Tensor *keras_Y = tensor_new(data_ctx, 3);
  for (int i = 0; i < 4; i++)
    tensor_append(data_ctx, keras_Y, y_data[i]);

  printf("Training with Keras API\n");
  printf("Learning rate: 0.5\n\n");

  for (int epoch = 0; epoch < 100; epoch++) {
    // Forward pass
    Tensor *output = dense_forward(keras_ctx, keras_X);
    if (!output) {
      printf("Forward pass failed\n");
      break;
    }

    // Compute loss gradient (softmax + cross-entropy combined)
    Tensor *loss_grad = cross_entropy_backward(data_ctx, output, keras_Y);
    if (!loss_grad) {
      printf("Loss gradient computation failed\n");
      break;
    }

    // Compute loss for display
    // Make a copy of output before computing cross_entropy (which expects
    // logits) Actually, we need pre-softmax logits for cross_entropy, but we
    // only have softmax output Let's just display the epoch
    if (epoch % 10 == 0 || epoch == 99) {
      printf("Epoch %3d completed\n", epoch);
    }

    // Backward pass
    int result = dense_backward(keras_ctx, 0.5f, loss_grad);
    if (result != 0) {
      printf("Backward pass failed with error: %d\n", result);
      break;
    }
  }

  // Final predictions
  printf("\nFinal predictions (Keras API):\n");
  Tensor *p = predict(keras_ctx, keras_X);
  if (!p) {
    printf("Prediction failed\n");
    return 1;
  }
  printf("Predictions:\n");
  tensor_print(p, -1, false);

  float acc = accuracy(keras_ctx, p, keras_Y);
  printf("Accuracy: %.3f\n", acc);

  dense_free(keras_ctx);
  tensor_free(data_ctx);

  return 0;
}
