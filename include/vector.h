#ifndef CTORCH_VECTOR_H
#define CTORCH_VECTOR_H

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#define len(src) (sizeof(src) / sizeof(src[0]))

typedef struct {
  size_t rows;
  size_t cols;
  size_t capacity;
  double *data;
} Vector;

typedef struct VectorContext VectorContext;

VectorContext *vector_create(void);
Vector *vector_new(VectorContext *ctx, const int column_count);
Vector *vector_new_tmp(const int column_count);

void vector_print(Vector *src, int count, bool suffle);
void vector_free(VectorContext *src);
void vector_free_tmp(Vector *v);

void vector_append(VectorContext *ctx, Vector *dest, const double *row_data);
void vector_append_tmp(Vector *dest, const double *row_data);

void vector_append_all(VectorContext *ctx, Vector *dest,
                       const double **row_data, size_t size);
double vector_get(const Vector *src, size_t row, size_t col);

Vector *vector_randn(VectorContext *ctx, size_t rows, size_t cols);

void vector_transpose(Vector *src);

Vector *vector_get_column(VectorContext *ctx, Vector *src, size_t col);
Vector *vector_remove_column(VectorContext *ctx, Vector *src, size_t col);

#endif // CTORCH_VECTOR_H
