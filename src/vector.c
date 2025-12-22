#define ARENA_IMPLEMENTATION
#define RAND_IMPLEMENTATION
#include "vector.h"
#include "arena.h"
#include "randn.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_ARENA_SIZE 1024 * 16

struct VectorContext {
  Arena *arena;
};

// Short lived functions that need to free immediately

Vector *vector_new_tmp(const int column_count) {
  Vector *v = calloc(1, sizeof(Vector));
  v->rows = 0;
  v->cols = column_count;
  v->capacity = 0;
  v->data = NULL;

  return v;
}

void vector_free_tmp(Vector *v) {
  if (!v)
    return;

  if (v->data)
    free(v->data);
  free(v);
}

void vector_append_tmp(Vector *dest, const double *row_data) {
  if (!dest || !row_data)
    return;

  if ((size_t)dest->rows == dest->capacity) {
    size_t new_cap = dest->capacity ? dest->capacity * 2 : 4;
    double *tmp = realloc(dest->data, new_cap * dest->cols * sizeof *tmp);
    if (!tmp) {
      fprintf(stderr, "Buy more ram LOL\n");
      exit(1);
    }
    dest->data = tmp;
    dest->capacity = new_cap;
  }
  memcpy(dest->data + dest->rows * dest->cols, row_data,
         dest->cols * sizeof *row_data);
  dest->rows++;
}

// Public functions

VectorContext *vector_create(void) {
  Arena *arena = arena_create(VECTOR_ARENA_SIZE);
  VectorContext *ctx = calloc(1, sizeof(VectorContext));
  ctx->arena = arena;
  return ctx;
}

Vector *vector_new(VectorContext *ctx, const int column_count) {
  Vector *v = arena_alloc(ctx->arena, sizeof(Vector), ARENA_ALIGNOF(Vector));
  v->rows = 0;
  v->cols = column_count;
  v->capacity = 0;
  v->data = NULL;

  return v;
}

void vector_print(Vector *src, int count, bool suffle) {
  if (!src)
    return;

  size_t rows = count <= 0 ? src->rows : count;

  printf("[");
  for (size_t i = 0; i < rows; i++) {
    printf("[");
    for (size_t j = 0; j < src->cols; j++) {
      int ii = suffle ? random_to(src->rows) : i;
      int jj = suffle ? random_to(src->cols) : j;

      double val = vector_get(src, ii, jj);
      if (j == src->cols - 1)
        printf("%f", val);
      else
        printf("%f, ", val);
    }
    if (i != rows - 1)
      printf("],\n");
    else
      printf("]");
  }
  printf("]\n");
}

void vector_free(VectorContext *ctx) {
  if (!ctx)
    return;

  arena_free(ctx->arena);
  free(ctx);
}

void vector_append(VectorContext *ctx, Vector *dest, const double *row_data) {
  if (!dest || !row_data)
    return;

  if ((size_t)dest->rows == dest->capacity) {
    size_t new_cap = dest->capacity ? dest->capacity * 2 : 4;
    double *tmp = arena_alloc(ctx->arena, new_cap * dest->cols * sizeof *tmp,
                              ARENA_ALIGNOF(double));
    if (!tmp) {
      fprintf(stderr, "Buy more ram LOL\n");
      exit(1);
    }
    memcpy(tmp, dest->data, dest->rows * dest->cols * sizeof *dest->data);
    dest->data = tmp;
    dest->capacity = new_cap;
  }
  memcpy(dest->data + dest->rows * dest->cols, row_data,
         dest->cols * sizeof *row_data);
  dest->rows++;
}

void vector_append_all(VectorContext *ctx, Vector *dest,
                       const double **row_data, size_t size) {
  if (!dest || !row_data)
    return;

  for (size_t i = 0; i < size; i++) {
    vector_append(ctx, dest, row_data[i]);
  }
}

double vector_get(const Vector *src, size_t row, size_t col) {
  return src->data[row * src->cols + col];
}

void vector_transpose(Vector *src) {
  if (!src)
    return;

  Vector *tmp = vector_new_tmp(src->rows);
  if (!tmp)
    return;

  for (size_t i = 0; i < src->cols; i++) {
    double rows[src->rows];
    memset(rows, 0, sizeof(rows));

    for (size_t j = 0; j < src->rows; j++) {
      rows[j] = vector_get(src, j, i);
    }
    vector_append_tmp(tmp, rows);
  }

  // Copy data
  memcpy(src->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  src->rows = tmp->rows;
  src->cols = tmp->cols;
  src->capacity = tmp->capacity;
  vector_free_tmp(tmp);
}

Vector *vector_randn(VectorContext *ctx, size_t rows, size_t cols) {
  if (!rows || !cols)
    return NULL;

  Vector *v = vector_new(ctx, cols);
  if (!v)
    return NULL;

  for (size_t i = 0; i < rows; i++) {
    double tmp[cols];
    memset(tmp, 0, sizeof(tmp));

    for (size_t j = 0; j < cols; j++) {
      tmp[j] = randn();
    }
    vector_append(ctx, v, tmp);
  }

  return v;
}

Vector *vector_get_column(VectorContext *ctx, Vector *src, size_t col) {
  if (!src)
    return NULL;

  Vector *v = vector_new(ctx, 1);
  if (!v)
    return NULL;

  for (size_t i = 0; i < src->rows; i++) {
    double tmp[1] = {vector_get(src, i, col)};
    vector_append(ctx, v, tmp);
  }

  return v;
}

Vector *vector_remove_column(VectorContext *ctx, Vector *src, size_t col) {
  if (!src || !ctx)
    return NULL;

  Vector *v = vector_new(ctx, src->cols - 1);
  if (!v)
    return NULL;

  for (size_t i = 0; i < src->rows; i++) {
    double tmp[src->cols - 1];
    memset(tmp, 0, sizeof(tmp));

    size_t dest_idx = 0;
    for (size_t j = 0; j < src->cols; j++) {
      if (j == col)
        continue;

      tmp[dest_idx++] = vector_get(src, i, j);
    }
    vector_append(ctx, v, tmp);
  }

  return v;
}
