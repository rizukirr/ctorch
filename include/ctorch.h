#ifndef CTORCH_H
#define CTORCH_H

#include "vector.h"

Vector *affine_transform(VectorContext *ctx, Vector *inputs, Vector *weights,
                         double *bias);

int activation_ReLU(Vector *inputs);

int activation_sigmoid(Vector *inputs);

int activation_softmax(Vector *inputs);

double cross_entropy(Vector *logits, Vector *labels);

#endif // CTORCH_H
