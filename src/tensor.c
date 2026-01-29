#define ARENA_IMPLEMENTATION
#define RAND_IMPLEMENTATION
#include "tensor.h"
#include "arena.h"
#include "errors.h"
#include "randn.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>

#define VECTOR_ARENA_SIZE 1024 * 16

struct TensorContext {
  Arena *arena;
};

// Short lived functions that need to free immediately

Tensor *tensor_create_tmp(const int num_cols) {
  if (num_cols <= 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "column count must be positive (received: %d)",
                         num_cols);
    return NULL;
  }

  Tensor *tensor = calloc(1, sizeof(Tensor));
  if (!tensor) {
    ctorch_set_error(CTORCH_ERROR_OUT_OF_MEMORY,
                     "failed to allocate temporary tensor structure");
    return NULL;
  }

  tensor->rows = 0;
  tensor->cols = num_cols;
  tensor->capacity = 0;
  tensor->data = NULL;

  return tensor;
}

void tensor_free_tmp(Tensor *tensor) {
  if (!tensor)
    return;

  if (tensor->data)
    free(tensor->data);
  free(tensor);
}

void tensor_append_tmp(Tensor *tensor, const float *values) {
  if (!tensor || !values)
    return;

  if ((size_t)tensor->rows == tensor->capacity) {
    size_t new_cap = tensor->capacity ? tensor->capacity * 2 : 4;
    float *tmp = realloc(tensor->data, new_cap * tensor->cols * sizeof *tmp);
    if (!tmp) {
      ctorch_set_error_fmt(
          CTORCH_ERROR_OUT_OF_MEMORY,
          "failed to grow temporary tensor capacity (current: %zu rows, "
          "attempting: %zu rows)",
          tensor->capacity, new_cap);
      return;
    }
    tensor->data = tmp;
    tensor->capacity = new_cap;
  }
  memcpy(tensor->data + tensor->rows * tensor->cols, values,
         tensor->cols * sizeof *values);
  tensor->rows++;
}

// Public functions

// Helper function for creating random normal tensors with scaling
static Tensor *tensor_randn_scaled(TensorContext *ctx, size_t rows, size_t cols,
                                   float scale) {
  if (!ctx) {
    return NULL;
  }

  if (!rows || !cols) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "dimensions must be positive (received: %zux%zu)",
                         rows, cols);
    return NULL;
  }

  Tensor *v = tensor_create(ctx, cols);
  if (!v)
    return NULL;

  for (size_t i = 0; i < rows; i++) {
    float tmp[cols];

    for (size_t j = 0; j < cols; j++) {
      tmp[j] = randn() * scale;
    }
    tensor_append(ctx, v, tmp);
  }

  return v;
}

TensorContext *tensor_init(void) {
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

Tensor *tensor_create(TensorContext *ctx, const int num_cols) {
  if (!ctx) {
    return NULL;
  }

  if (num_cols <= 0) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "column count must be positive (received: %d)",
                         num_cols);
    return NULL;
  }

  Tensor *tensor =
      arena_alloc(ctx->arena, sizeof(Tensor), ARENA_ALIGNOF(Tensor));
  if (!tensor) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
        "failed to allocate tensor structure (requested columns: %d)",
        num_cols);
    return NULL;
  }

  tensor->rows = 0;
  tensor->cols = num_cols;
  tensor->capacity = 0;
  tensor->data = NULL;

  return tensor;
}

void tensor_print(Tensor *tensor, int max_rows, bool shuffle) {
  if (!tensor)
    return;

  size_t rows = (max_rows <= 0 || (size_t)max_rows > tensor->rows)
                    ? tensor->rows
                    : (size_t)max_rows;

  printf("[\n");

  for (size_t i = 0; i < rows; i++) {
    size_t ii = shuffle ? random_to(tensor->rows) : i;

    printf("  [");

    for (size_t j = 0; j < tensor->cols; j++) {
      float val = tensor->data[ii * tensor->cols + j];

      printf("%f", val);
      if (j + 1 < tensor->cols)
        printf(", ");
    }

    printf("]");
    if (i + 1 < rows)
      printf(",\n");
  }

  printf("\n]\n");
}

void tensor_free(TensorContext *ctx) {
  if (!ctx)
    return;

  if (ctx->arena)
    arena_free(ctx->arena);
  free(ctx);
}

void tensor_append(TensorContext *ctx, Tensor *tensor, const float *values) {
  if (!ctx) {
    return;
  }

  if (!tensor) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return;
  }

  if (!values) {
    ctorch_set_error_fmt(CTORCH_ERROR_NULL_PARAMETER,
                         "values is NULL (tensor shape: %zux%zu)", tensor->rows,
                         tensor->cols);
    return;
  }

  if ((size_t)tensor->rows == tensor->capacity) {
    size_t new_cap = tensor->capacity ? tensor->capacity * 2 : 4;
    float *tmp = arena_alloc(ctx->arena, new_cap * tensor->cols * sizeof *tmp,
                             ARENA_ALIGNOF(float));
    if (!tmp) {
      ctorch_set_error_fmt(CTORCH_ERROR_ARENA_ALLOCATION_FAILED,
                           "failed to grow tensor capacity (current: %zu rows, "
                           "attempting: %zu rows)",
                           tensor->capacity, new_cap);
      return;
    }
    memcpy(tmp, tensor->data,
           tensor->rows * tensor->cols * sizeof *tensor->data);
    tensor->data = tmp;
    tensor->capacity = new_cap;
  }
  memcpy(tensor->data + tensor->rows * tensor->cols, values,
         tensor->cols * sizeof *values);
  tensor->rows++;
}

void tensor_append_all(TensorContext *ctx, Tensor *tensor, const float **values,
                       size_t num_rows) {
  if (!ctx) {
    return;
  }

  if (!tensor) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return;
  }

  if (!values) {
    ctorch_set_error_fmt(CTORCH_ERROR_NULL_PARAMETER,
                         "values array is NULL (attempting to append %zu rows)",
                         num_rows);
    return;
  }

  for (size_t i = 0; i < num_rows; i++) {
    tensor_append(ctx, tensor, values[i]);
  }
}

float tensor_get(const Tensor *tensor, size_t row, size_t col) {
  if (!tensor)
    return 0.0;

  if (row >= tensor->rows)
    return 0.0;

  if (col >= tensor->cols)
    return 0.0;

  return tensor->data[row * tensor->cols + col];
}

void tensor_transpose(Tensor *tensor) {
  if (!tensor)
    return;

  Tensor *tmp = tensor_create_tmp(tensor->rows);
  if (!tmp)
    return;

  for (size_t i = 0; i < tensor->cols; i++) {
    float row_values[tensor->rows];

    for (size_t j = 0; j < tensor->rows; j++) {
      row_values[j] = tensor->data[j * tensor->cols + i];
    }
    tensor_append_tmp(tmp, row_values);
  }

  memcpy(tensor->data, tmp->data, tmp->rows * tmp->cols * sizeof *tmp->data);
  tensor->rows = tmp->rows;
  tensor->cols = tmp->cols;

  tensor_free_tmp(tmp);
}

Tensor *tensor_randn(TensorContext *ctx, size_t rows, size_t cols) {
  return tensor_randn_scaled(ctx, rows, cols, 0.01f);
}

Tensor *tensor_randn_he(TensorContext *ctx, size_t rows, size_t cols) {
  if (!rows) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "dimensions must be positive (received: %zux%zu)",
                         rows, cols);
    return NULL;
  }
  float scale = sqrtf(2.0f / (float)rows);
  return tensor_randn_scaled(ctx, rows, cols, scale);
}

Tensor *tensor_randn_xavier(TensorContext *ctx, size_t rows, size_t cols) {
  if (!rows || !cols) {
    ctorch_set_error_fmt(CTORCH_ERROR_INVALID_SHAPE,
                         "dimensions must be positive (received: %zux%zu)",
                         rows, cols);
    return NULL;
  }
  float scale = sqrtf(2.0f / (float)(rows + cols));
  return tensor_randn_scaled(ctx, rows, cols, scale);
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

  if (axis == AxisColumn && index >= src->cols) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_OUT_OF_BOUNDS,
        "column index out of bounds (index: %zu, valid range: 0-%zu)", index,
        src->cols - 1);
    return NULL;
  }

  size_t col_size = axis == AxisColumn ? 1 : src->cols;
  Tensor *v = tensor_create(ctx, col_size);
  if (!v)
    return NULL;

  for (size_t i = 0; i < src->rows; i++) {
    if (axis == AxisRow && i != index)
      continue;

    size_t tmp_size = axis == AxisRow ? src->cols : 1;
    float tmp[tmp_size];

    if (axis == AxisRow) {
      for (size_t j = 0; j < src->cols; j++) {
        tmp[j] = src->data[i * src->cols + j];
      }
    } else {
      tmp[0] = src->data[i * src->cols + index];
    }

    tensor_append(ctx, v, tmp);
  }

  return v;
}

int tensor_copy(Tensor *src, Tensor *dest) {
  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "source tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (!dest) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "destination tensor is NULL");
    return CTORCH_ERROR_NULL_PARAMETER;
  }

  if (src->rows != dest->rows || src->cols != dest->cols) {
    ctorch_set_error_fmt(CTORCH_ERROR_DIMENSION_MISMATCH,
                         "source and destination tensors must have the same "
                         "dimensions (source: %zux%zu, destination: %zux%zu)",
                         src->rows, src->cols, dest->rows, dest->cols);
    return CTORCH_ERROR_DIMENSION_MISMATCH;
  }

  memcpy(dest->data, src->data, src->rows * src->cols * sizeof *src->data);

  return 0;
}

Tensor *tensor_dup(TensorContext *ctx, Tensor *src) {
  if (!ctx) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "context is NULL");
    return NULL;
  }

  if (!src) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "source tensor is NULL");
    return NULL;
  }

  Tensor *dest = tensor_create(ctx, src->cols);
  if (!dest) {
    return NULL;
  }

  for (size_t i = 0; i < src->rows; i++) {
    tensor_append(ctx, dest, &src->data[i * src->cols]);
  }

  return dest;
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

  if (axis == AxisColumn && index >= src->cols) {
    ctorch_set_error_fmt(
        CTORCH_ERROR_OUT_OF_BOUNDS,
        "column index out of bounds (index: %zu, valid range: 0-%zu)", index,
        src->cols - 1);
    return NULL;
  }

  size_t col_size = axis == AxisColumn ? src->cols - 1 : src->cols;
  Tensor *v = tensor_create(ctx, col_size);
  if (!v)
    return NULL;

  for (size_t i = 0; i < src->rows; i++) {
    if (axis == AxisRow && i == index)
      continue;

    float tmp[col_size];
    size_t dest_idx = 0;

    for (size_t j = 0; j < src->cols; j++) {
      if (axis == AxisColumn && j == index)
        continue;

      tmp[dest_idx++] = src->data[i * src->cols + j];
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
  } else { // axis == AxisColumn
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
      result[i] = src->data[i * src->cols + index];
    }
    return result;
  }
}

float tensor_sum(Tensor *tensor) {
  if (!tensor) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return NAN;
  }

  if (!tensor->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "tensor data is NULL");
    return NAN;
  }

  float sum = 0.0f;
  size_t total_elements = tensor->rows * tensor->cols;
  for (size_t i = 0; i < total_elements; i++) {
    sum += tensor->data[i];
  }
  return sum;
}

float tensor_avg(Tensor *tensor) {
  if (!tensor) {
    ctorch_set_error(CTORCH_ERROR_NULL_PARAMETER, "tensor is NULL");
    return NAN;
  }

  if (!tensor->data) {
    ctorch_set_error(CTORCH_ERROR_NULL_DATA, "tensor data is NULL");
    return NAN;
  }

  if (tensor->rows == 0 || tensor->cols == 0) {
    ctorch_set_error(CTORCH_ERROR_INVALID_SHAPE, "tensor has zero dimensions");
    return NAN;
  }

  float sum = tensor_sum(tensor);
  if (isnan(sum)) {
    return NAN;
  }
  return sum / (tensor->rows * tensor->cols);
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

  Tensor *v = tensor_create(ctx, cols);
  if (!v)
    return NULL;

  for (size_t i = 0; i < rows; i++) {
    float tmp[cols];
    memset(tmp, 0, sizeof(tmp));
    tensor_append(ctx, v, tmp);
  }

  return v;
}
