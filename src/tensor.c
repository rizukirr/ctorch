#define ARENA_IMPLEMENTATION
#define RAND_IMPLEMENTATION
#include "tensor.h"
#include "arena.h"
#include "randn.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_ARENA_SIZE 1024 * 16

struct TensorContext {
  Arena *arena;
};

// Short lived functions that need to free immediately

Tensor *tensor_new_tmp(const int column_count) {
  if (column_count <= 0) {
    return NULL;
  }

  Tensor *v = calloc(1, sizeof(Tensor));
  if (!v) {
    return NULL;
  }

  v->rows = 0;
  v->cols = column_count;
  v->capacity = 0;
  v->data = NULL;

  return v;
}

void tensor_free_tmp(Tensor *v) {
  if (!v)
    return;

  if (v->data)
    free(v->data);
  free(v);
}

void tensor_append_tmp(Tensor *dest, const float *row_data) {
  if (!dest || !row_data)
    return;

  if ((size_t)dest->rows == dest->capacity) {
    size_t new_cap = dest->capacity ? dest->capacity * 2 : 4;
    float *tmp = realloc(dest->data, new_cap * dest->cols * sizeof *tmp);
    if (!tmp) {
      // Cannot set error for tmp tensors - they don't have a context
      return;
    }
    dest->data = tmp;
    dest->capacity = new_cap;
  }
  memcpy(dest->data + dest->rows * dest->cols, row_data,
         dest->cols * sizeof *row_data);
  dest->rows++;
}

// Public functions

TensorContext *tensor_create(void) {
  Arena *arena = arena_create(VECTOR_ARENA_SIZE);
  if (!arena)
    return NULL;

  TensorContext *ctx = calloc(1, sizeof(TensorContext));
  if (!ctx) {
    arena_free(arena);
    return NULL;
  }

  ctx->arena = arena;
  return ctx;
}

Tensor *tensor_new(TensorContext *ctx, const int column_count) {
  if (!ctx) {
    return NULL;
  }

  if (column_count <= 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "column count must be positive (received: %d)",
                         column_count);
    return NULL;
  }

  Tensor *v = arena_alloc(ctx->arena, sizeof(Tensor), ARENA_ALIGNOF(Tensor));
  if (!v) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
        "failed to allocate tensor structure (requested columns: %d)",
        column_count);
    return NULL;
  }

  v->rows = 0;
  v->cols = column_count;
  v->capacity = 0;
  v->data = NULL;

  return v;
}

void tensor_print(Tensor *src, int count, bool shuffle) {
  if (!src)
    return;

  size_t rows = count <= 0 ? src->rows : count;

  printf("[");
  for (size_t i = 0; i < rows; i++) {
    printf("[");
    for (size_t j = 0; j < src->cols; j++) {
      int ii = shuffle ? random_to(src->rows) : i;
      int jj = shuffle ? random_to(src->cols) : j;

      float val = tensor_get(src, ii, jj);
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

void tensor_free(TensorContext *ctx) {
  if (!ctx)
    return;

  arena_free(ctx->arena);
  free(ctx);
}

void tensor_append(TensorContext *ctx, Tensor *dest, const float *row_data) {
  if (!ctx) {
    return;
  }

  if (!dest) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "destination tensor is NULL");
    return;
  }

  if (!row_data) {
    ctorch_set_error_fmt(CTORCH_ERROR_NULL_PARAMETER,
                         "row data is NULL (tensor shape: %zux%zu)", dest->rows,
                         dest->cols);
    return;
  }

  if ((size_t)dest->rows == dest->capacity) {
    size_t new_cap = dest->capacity ? dest->capacity * 2 : 4;
    float *tmp = arena_alloc(ctx->arena, new_cap * dest->cols * sizeof *tmp,
                             ARENA_ALIGNOF(float));
    if (!tmp) {
      ctorch_set_error_fmt(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                           "failed to grow tensor capacity (current: %zu rows, "
                           "attempting: %zu rows)",
                           dest->capacity, new_cap);
      return;
    }
    memcpy(tmp, dest->data, dest->rows * dest->cols * sizeof *dest->data);
    dest->data = tmp;
    dest->capacity = new_cap;
  }
  memcpy(dest->data + dest->rows * dest->cols, row_data,
         dest->cols * sizeof *row_data);
  dest->rows++;
}

void tensor_append_all(TensorContext *ctx, Tensor *dest, const float **row_data,
                       size_t size) {
  if (!ctx) {
    return;
  }

  if (!dest) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "destination tensor is NULL");
    return;
  }

  if (!row_data) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_NULL_PARAMETER,
        "row data array is NULL (attempting to append %zu rows)", size);
    return;
  }

  for (size_t i = 0; i < size; i++) {
    tensor_append(ctx, dest, row_data[i]);
  }
}

float tensor_get(const Tensor *src, size_t row, size_t col) {
  if (!src)
    return 0.0;

  if (row >= src->rows)
    return 0.0;

  if (col >= src->cols)
    return 0.0;

  return src->data[row * src->cols + col];
}

void tensor_transpose(Tensor *src) {
  if (!src)
    return;

  Tensor *tmp = tensor_new_tmp(src->rows);
  if (!tmp)
    return;

  for (size_t i = 0; i < src->cols; i++) {
    float rows[src->rows];
    memset(rows, 0, sizeof(rows));

    for (size_t j = 0; j < src->rows; j++) {
      rows[j] = tensor_get(src, j, i);
    }
    tensor_append_tmp(tmp, rows);
  }

  // Copy data
  memcpy(src->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  src->rows = tmp->rows;
  src->cols = tmp->cols;
  src->capacity = tmp->capacity;
  tensor_free_tmp(tmp);
}

Tensor *tensor_randn(TensorContext *ctx, size_t rows, size_t cols) {
  if (!ctx) {
    return NULL;
  }

  if (!rows || !cols) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "dimensions must be positive (received: %zux%zu)",
                         rows, cols);
    return NULL;
  }

  Tensor *v = tensor_new(ctx, cols);
  if (!v)
    return NULL;

  for (size_t i = 0; i < rows; i++) {
    float tmp[cols];
    memset(tmp, 0, sizeof(tmp));

    for (size_t j = 0; j < cols; j++) {
      tmp[j] = randn();
    }
    tensor_append(ctx, v, tmp);
  }

  return v;
}

Tensor *tensor_select(TensorContext *ctx, Tensor *src, size_t index,
                      Axis axis) {
  if (!ctx) {
    return NULL;
  }

  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "source tensor is NULL");
    return NULL;
  }

  if (axis == AxisRow && index >= src->rows) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_OUT_OF_BOUNDS,
        "row index out of bounds (index: %zu, valid range: 0-%zu)", index,
        src->rows - 1);
    return NULL;
  }

  if (axis == AxisColum && index >= src->cols) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_OUT_OF_BOUNDS,
        "column index out of bounds (index: %zu, valid range: 0-%zu)", index,
        src->cols - 1);
    return NULL;
  }

  size_t col_size = axis == AxisColum ? 1 : src->cols;
  Tensor *v = tensor_new(ctx, col_size);
  if (!v)
    return NULL;

  for (size_t i = 0; i < src->rows; i++) {
    if (axis == AxisRow && i != index)
      continue;

    size_t tmp_size = axis == AxisRow ? src->cols : 1;
    float tmp[tmp_size];

    if (axis == AxisRow) {
      for (size_t j = 0; j < src->cols; j++) {
        tmp[j] = tensor_get(src, i, j);
      }
    } else {
      tmp[0] = tensor_get(src, i, index);
    }

    tensor_append(ctx, v, tmp);
  }

  return v;
}

Tensor *tensor_drop(TensorContext *ctx, Tensor *src, size_t index, Axis axis) {
  if (!ctx) {
    return NULL;
  }

  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "source tensor is NULL");
    return NULL;
  }

  if (axis == AxisRow && index >= src->rows) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_OUT_OF_BOUNDS,
        "row index out of bounds (index: %zu, valid range: 0-%zu)", index,
        src->rows - 1);
    return NULL;
  }

  if (axis == AxisColum && index >= src->cols) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_OUT_OF_BOUNDS,
        "column index out of bounds (index: %zu, valid range: 0-%zu)", index,
        src->cols - 1);
    return NULL;
  }

  size_t col_size = axis == AxisColum ? src->cols - 1 : src->cols;
  Tensor *v = tensor_new(ctx, col_size);
  if (!v)
    return NULL;

  for (size_t i = 0; i < src->rows; i++) {
    if (axis == AxisRow && i == index)
      continue;

    float tmp[col_size];
    memset(tmp, 0, sizeof(tmp));
    size_t dest_idx = 0;

    for (size_t j = 0; j < src->cols; j++) {
      if (axis == AxisColum && j == index)
        continue;

      tmp[dest_idx++] = tensor_get(src, i, j);
    }
    tensor_append(ctx, v, tmp);
  }

  return v;
}

float *tensor_slice(TensorContext *ctx, Tensor *src, size_t index, Axis axis) {
  if (!ctx) {
    return NULL;
  }

  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "source tensor is NULL");
    return NULL;
  }

  if (axis == AxisRow) {
    if (index >= src->rows) {
      ctorch_set_error_fmt(
          CTORCH_ERROR_OUT_OF_BOUNDS,
          "row index out of bounds (index: %zu, valid range: 0-%zu)", index,
          src->rows - 1);
      return NULL;
    }

    float *result = arena_alloc(ctx->arena, src->cols * sizeof(float),
                                ARENA_ALIGNOF(float));
    if (!result) {
      ctorch_set_error_fmt(
          CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
          "failed to allocate slice buffer (requested: %zu floats)", src->cols);
      return NULL;
    }

    memcpy(result, src->data + index * src->cols, src->cols * sizeof(float));
    return result;
  } else { // axis == AxisColum
    if (index >= src->cols) {
      ctorch_set_error_fmt(
          CTORCH_ERROR_OUT_OF_BOUNDS,
          "column index out of bounds (index: %zu, valid range: 0-%zu)", index,
          src->cols - 1);
      return NULL;
    }

    float *result = arena_alloc(ctx->arena, src->rows * sizeof(float),
                                ARENA_ALIGNOF(float));
    if (!result) {
      ctorch_set_error_fmt(
          CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
          "failed to allocate slice buffer (requested: %zu floats)", src->rows);
      return NULL;
    }

    for (size_t i = 0; i < src->rows; i++) {
      result[i] = tensor_get(src, i, index);
    }
    return result;
  }
}
