#define ARENA_IMPLEMENTATION
#define RAND_IMPLEMENTATION
#include "tensor.h"
#include "arena.h"
#include "errors.h"
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
  v->grad = NULL;
  v->data = NULL;

  return v;
}

int tensor_free_tmp(Tensor *v) {
  if (!v) {
    return 0; // NULL is valid for free operations
  }

  if (v->data)
    free(v->data);
  free(v);
  return 0;
}

int tensor_append_tmp(Tensor *dest, const float *row_data) {
  if (!dest) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!row_data) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "row data is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if ((size_t)dest->rows == dest->capacity) {
    size_t new_cap = dest->capacity ? dest->capacity * 2 : 4;
    float *tmp = realloc(dest->data, new_cap * dest->cols * sizeof *tmp);
    if (!tmp) {
      ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                       "failed to grow temporary tensor capacity");
      return CTORCH_ERROR_OUT_OF_MEMORY;
    }
    dest->data = tmp;
    dest->capacity = new_cap;
  }
  memcpy(dest->data + dest->rows * dest->cols, row_data,
         dest->cols * sizeof *row_data);
  dest->rows++;
  return 0;
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
  v->grad = NULL;
  v->data = NULL;

  return v;
}

int tensor_print(Tensor *src, int count, bool shuffle) {
  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  size_t rows = count <= 0 ? src->rows : count;

  if (printf("[") < 0) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY, "output operation failed");
    return CTORCH_ERROR_OUT_OF_MEMORY;
  }

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
  return 0;
}

int tensor_free(TensorContext *ctx) {
  if (!ctx) {
    return 0; // NULL is valid for free operations
  }

  arena_free(ctx->arena);
  free(ctx);
  return 0;
}

int tensor_append(TensorContext *ctx, Tensor *dest, const float *row_data) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!dest) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "destination tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!row_data) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "row data is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if ((size_t)dest->rows == dest->capacity) {
    size_t new_cap = dest->capacity ? dest->capacity * 2 : 4;
    float *tmp = arena_alloc(ctx->arena, new_cap * dest->cols * sizeof *tmp,
                             ARENA_ALIGNOF(float));
    if (!tmp) {
      ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                       "failed to grow tensor capacity");
      return CTORCH_ERROR_ARENA_ALLOCATION_FAILED;
    }
    memcpy(tmp, dest->data, dest->rows * dest->cols * sizeof *dest->data);
    dest->data = tmp;
    dest->capacity = new_cap;
  }
  memcpy(dest->data + dest->rows * dest->cols, row_data,
         dest->cols * sizeof *row_data);
  dest->rows++;
  return 0;
}

int tensor_append_all(TensorContext *ctx, Tensor *dest, const float **row_data,
                      size_t size) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!dest) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "destination tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!row_data) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "row data array is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  for (size_t i = 0; i < size; i++) {
    int ret = tensor_append(ctx, dest, row_data[i]);
    if (ret < 0) {
      return ret; // Propagate error
    }
  }
  return 0;
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

int tensor_transpose(Tensor *src) {
  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  Tensor *tmp = tensor_new_tmp(src->rows);
  if (!tmp) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor for transpose");
    return CTORCH_ERROR_OUT_OF_MEMORY;
  }

  for (size_t i = 0; i < src->cols; i++) {
    float rows[src->rows];
    memset(rows, 0, sizeof(rows));

    for (size_t j = 0; j < src->rows; j++) {
      rows[j] = tensor_get(src, j, i);
    }
    int ret = tensor_append_tmp(tmp, rows);
    if (ret < 0) {
      tensor_free_tmp(tmp);
      return ret; // Propagate error
    }
  }

  // Copy data
  memcpy(src->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  src->rows = tmp->rows;
  src->cols = tmp->cols;
  src->capacity = tmp->capacity;
  tensor_free_tmp(tmp);
  return 0;
}

float *scalar_randn(TensorContext *ctx, size_t size) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!size) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "size must be positive (received: %zu)", size);
    return NULL;
  }

  float *v =
      arena_alloc(ctx->arena, size * sizeof(float), ARENA_ALIGNOF(float));
  if (!v) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
        "failed to allocate scalar (requested size: %zu bytes)", size);
    return NULL;
  }

  for (size_t i = 0; i < size; i++) {
    v[i] = randn();
  }

  return v;
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

Tensor *tensor_zeros(TensorContext *ctx, size_t rows, size_t cols) {
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
      tmp[j] = 0.0;
    }
    tensor_append(ctx, v, tmp);
  }

  return v;
}

Tensor *tensor_sum_rows(TensorContext *ctx, Tensor *src) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "source tensor is NULL");
    return NULL;
  }

  Tensor *v = tensor_new(ctx, 1);
  if (!v)
    return NULL;

  for (size_t i = 0; i < src->rows; i++) {
    float sum = 0;
    for (size_t j = 0; j < src->cols; j++) {
      sum += tensor_get(src, i, j);
    }
    float row_sum[] = {sum};
    tensor_append(ctx, v, row_sum);
  }

  return v;
}

Tensor *tensor_mul_elementwise(TensorContext *ctx, Tensor *a, Tensor *b) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!a || !b) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER,
                     !a ? "tensor a is NULL" : "tensor b is NULL");
    return NULL;
  }

  if (a->rows != b->rows || a->cols != b->cols) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "dimension mismatch (a: %zux%zu, b: %zux%zu) - "
                         "expected same shape",
                         a->rows, a->cols, b->rows, b->cols);
    return NULL;
  }

  Tensor *v = tensor_new(ctx, a->cols);
  if (!v)
    return NULL;

  for (size_t i = 0; i < a->rows; i++) {
    float row_data[a->cols];
    for (size_t j = 0; j < a->cols; j++) {
      row_data[j] = tensor_get(a, i, j) * tensor_get(b, i, j);
    }
    tensor_append(ctx, v, row_data);
  }

  return v;
}

Tensor *tensor_copy(TensorContext *ctx, Tensor *src) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "source tensor is NULL");
    return NULL;
  }

  Tensor *copy = tensor_new(ctx, src->cols);
  if (!copy)
    return NULL;

  // Allocate capacity to match source tensor
  if (src->rows > 0) {
    size_t total_size = src->rows * src->cols;
    copy->data = arena_alloc(ctx->arena, total_size * sizeof(float),
                             ARENA_ALIGNOF(float));
    if (!copy->data) {
      ctorch_set_error(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                       "failed to allocate tensor copy data");
      return NULL;
    }

    // Copy data
    memcpy(copy->data, src->data, total_size * sizeof(float));
    copy->rows = src->rows;
    copy->capacity = src->rows;
  }

  // Note: gradient is NOT copied (remains NULL)
  copy->grad = NULL;

  return copy;
}

int tensor_allocate_grad(TensorContext *ctx, Tensor *t) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!t) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  // If gradient already allocated, nothing to do
  if (t->grad)
    return 0;

  size_t grad_size = t->rows * t->cols;
  if (grad_size == 0)
    return 0; // Empty tensor, no gradient needed

  t->grad =
      arena_alloc(ctx->arena, grad_size * sizeof(float), ARENA_ALIGNOF(float));
  if (!t->grad) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
        "failed to allocate gradient array (requested: %zu floats)", grad_size);
    return CTORCH_ERROR_ARENA_ALLOCATION_FAILED;
  }

  // Initialize gradient to zero
  memset(t->grad, 0, grad_size * sizeof(float));

  return 0;
}

void tensor_zero_grad(Tensor *t) {
  if (!t || !t->grad)
    return;

  size_t grad_size = t->rows * t->cols;
  memset(t->grad, 0, grad_size * sizeof(float));
}
