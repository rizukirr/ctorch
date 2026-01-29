#ifndef CTORCH_TENSOR_H
#define CTORCH_TENSOR_H

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define len(src) (sizeof(src) / sizeof(src[0]))

#define array_append(arena, dest, data)                                        \
  do {                                                                         \
    if ((dest)->size == (dest)->capacity) {                                    \
      size_t new_cap = (dest)->capacity ? (dest)->capacity * 2 : 4;            \
      void *new_items =                                                        \
          arena_alloc((arena), new_cap * sizeof(*(dest)->items),               \
                      ARENA_ALIGNOF(__typeof__(*(dest)->items)));              \
      if (!new_items)                                                          \
        abort();                                                               \
      if ((dest)->items)                                                       \
        memcpy(new_items, (dest)->items,                                       \
               (dest)->size * sizeof(*(dest)->items));                         \
      (dest)->items = new_items;                                               \
      (dest)->capacity = new_cap;                                              \
    }                                                                          \
    (dest)->items[(dest)->size++] = (data);                                    \
  } while (0)

/**
 * @brief 2D matrix/tensor data structure.
 *
 * Represents a dynamically sized 2D array stored in row-major order.
 * Used as the primary tensor type for neural network operations.
 *
 * @field rows      Number of rows in the matrix
 * @field cols      Number of columns in the matrix
 * @field capacity  Allocated capacity for growth
 * @field data      Contiguous array storing matrix data in row-major order
 */
typedef struct {
  size_t rows;
  size_t cols;
  size_t capacity;
  float *data;
} Tensor;

/**
 * @brief Axis selection for tensor slicing and selection operations.
 *
 * Specifies whether operations should work along rows or columns:
 *   - AxisRow: Operate along rows (select/slice a specific row)
 *   - AxisColumn: Operate along columns (select/slice a specific column)
 *
 * Used by functions like tensor_select(), tensor_slice(), and tensor_drop().
 */
typedef enum { AxisRow, AxisColumn } Axis;

/**
 * @brief Opaque memory context for tensor allocation.
 *
 * Manages memory using an arena allocator for efficient batch allocation
 * and deallocation of tensors.
 */
typedef struct TensorContext TensorContext;

/**
 * @brief Creates a new tensor memory context.
 *
 * Initializes an arena-based memory context for managing tensor allocations.
 * All tensors created with this context will be freed when the context is
 * freed.
 *
 * @return Pointer to new TensorContext, or NULL on allocation failure
 */
TensorContext *tensor_init(void);

/**
 * @brief Creates a new tensor with specified column count.
 *
 * Allocates a new empty tensor using the arena allocator. The tensor starts
 * with 0 rows and will grow dynamically as data is appended.
 *
 * @param ctx       Memory context for allocation
 * @param num_cols  Number of columns in the tensor
 *
 * @return Pointer to new Tensor, or NULL on allocation failure
 */
Tensor *tensor_create(TensorContext *ctx, const int num_cols);

/**
 * @brief Creates a temporary tensor with manual memory management.
 *
 * Creates a tensor using standard malloc/free instead of arena allocation.
 * Must be freed with tensor_free_tmp(). Used for short-lived intermediate
 * results.
 *
 * @param num_cols  Number of columns in the tensor
 *
 * @return Pointer to new Tensor, or NULL on allocation failure
 */
Tensor *tensor_create_tmp(const int num_cols);

/**
 * @brief Prints tensor contents to stdout.
 *
 * Displays the tensor in matrix notation with square brackets.
 * Can optionally shuffle the displayed values for sampling large tensors.
 *
 * @param tensor    Tensor to print
 * @param max_rows  Maximum number of rows to print (0 or negative for all rows)
 * @param shuffle   If true, randomly samples elements instead of sequential
 *                  access
 */
void tensor_print(Tensor *tensor, int max_rows, bool shuffle);

/**
 * @brief Frees tensor context and all associated tensors.
 *
 * Releases all memory allocated through the arena allocator, including
 * all tensors created with this context.
 *
 * @param ctx  TensorContext to free
 */
void tensor_free(TensorContext *ctx);

/**
 * @brief Frees a temporary tensor.
 *
 * Releases memory for a tensor created with tensor_create_tmp().
 * Do not use this for tensors created with tensor_create().
 *
 * @param tensor  Tensor to free
 */
void tensor_free_tmp(Tensor *tensor);

/**
 * @brief Appends a row to the tensor.
 *
 * Adds a new row to the end of the tensor, automatically growing capacity
 * if needed (doubles capacity when full). Uses arena allocation.
 *
 * @param ctx     Memory context for allocation
 * @param tensor  Tensor to append to
 * @param values  Array of floats with length equal to tensor's column count
 */
void tensor_append(TensorContext *ctx, Tensor *tensor, const float *values);

/**
 * @brief Appends a row to a temporary tensor.
 *
 * Like tensor_append() but for temporary tensors created with
 * tensor_create_tmp(). Uses standard realloc for memory management.
 *
 * @param tensor  Temporary tensor to append to
 * @param values  Array of floats with length equal to tensor's column count
 */
void tensor_append_tmp(Tensor *tensor, const float *values);

/**
 * @brief Appends multiple rows to the tensor.
 *
 * Batch append operation that adds multiple rows in sequence.
 *
 * @param ctx       Memory context for allocation
 * @param tensor    Tensor to append to
 * @param values    Array of row pointers, each pointing to an array of floats
 * @param num_rows  Number of rows to append
 */
void tensor_append_all(TensorContext *ctx, Tensor *tensor, const float **values,
                       size_t num_rows);

/**
 * @brief Gets element at specified row and column.
 *
 * Accesses matrix element using row-major indexing.
 * No bounds checking in release builds.
 *
 * Math: element = data[row * cols + col]
 *
 * @param tensor  Tensor to access
 * @param row     Row index (0-based)
 * @param col     Column index (0-based)
 *
 * @return Value at the specified position
 */
float tensor_get(const Tensor *tensor, size_t row, size_t col);

/**
 * @brief Copies tensor data from source to destination.
 *
 * Copies all data from src tensor to dest tensor. Both tensors must have
 * the same dimensions. The destination tensor's data buffer is overwritten
 * with the source tensor's data.
 *
 * @param src   Source tensor to copy from
 * @param dest  Destination tensor to copy to
 *
 * @return 0 on success, negative CTorchError code on failure
 * @note Both tensors must already be allocated with matching dimensions
 */
int tensor_copy(Tensor *src, Tensor *dest);

/**
 * @brief Creates a tensor filled with random normal values.
 *
 * Generates a new tensor with values sampled from a standard normal
 * distribution (mean=0, std=1) using Box-Muller transform.
 *
 * Math: values ~ N(0, 1)
 *
 * @param ctx   Memory context for allocation
 * @param rows  Number of rows
 * @param cols  Number of columns
 *
 * @return Pointer to new tensor filled with random values, or NULL on error
 */
Tensor *tensor_randn(TensorContext *ctx, size_t rows, size_t cols);

/**
 * @brief Creates a tensor filled with random normal values with He
 * initialization.
 *
 * Generates a new tensor with values sampled from a standard normal
 * distribution (mean=0, std=1) using Box-Muller transform.
 * Uses He initialization: https://arxiv.org/abs/1502.01852
 *
 * Math: values ~ N(0, 1)
 *
 * @param ctx   Memory context for allocation
 * @param rows  Number of rows
 * @param cols  Number of columns
 *
 * @return Pointer to new tensor filled with random values, or NULL on error
 */
Tensor *tensor_randn_he(TensorContext *ctx, size_t rows, size_t cols);

/**
 * @brief Creates a tensor filled with random normal values with Xavier
 * initialization.
 *
 * Generates a new tensor with values sampled from a standard normal
 * distribution (mean=0, std=1) using Box-Muller transform.
 * Uses Xavier initialization: https://arxiv.org/abs/1704.08863
 *
 * Math: values ~ N(0, sqrt(2/(rows + cols)))
 *
 * @param ctx   Memory context for allocation
 * @param rows  Number of rows
 * @param cols  Number of columns
 *
 * @return Pointer to new tensor filled with random values, or NULL on error
 */
Tensor *tensor_randn_xavier(TensorContext *ctx, size_t rows, size_t cols);

/**
 * @brief Transposes the tensor in-place.
 *
 * Swaps rows and columns of the matrix. After transpose, a matrix of
 * shape (m, n) becomes (n, m).
 *
 * Math: B[j][i] = A[i][j] for all i, j
 *
 * @param tensor  Tensor to transpose (modified in-place)
 */
void tensor_transpose(Tensor *tensor);

/**
 * @brief Selects a single row or column from a tensor.
 *
 * Returns a new tensor containing only the specified row or column from
 * the source tensor. The result is a tensor with either 1 row (if selecting
 * a row) or 1 column (if selecting a column).
 *
 * Result shapes:
 *   - AxisRow: Returns tensor of shape (1, cols) containing the selected row
 *   - AxisColumn: Returns tensor of shape (rows, 1) containing the selected
 * column
 *
 * @param ctx    Memory context for allocation
 * @param src    Source tensor
 * @param index  Row or column index to select (0-based)
 * @param axis   Either AxisRow or AxisColumn to specify selection direction
 *
 * @return New tensor containing the selected row/column, or NULL on error
 */
Tensor *tensor_select(TensorContext *ctx, Tensor *src, size_t index, Axis axis);

/**
 * @brief Extracts a 1D slice from a 2D tensor.
 *
 * Returns a newly allocated 1D array containing either a row or column from
 * the source tensor. The array is allocated using the arena allocator and
 * will be freed when the context is freed.
 *
 * Result:
 *   - AxisRow axis: Returns array of length cols with data from row `index`
 *   - AxisColumn axis: Returns array of length rows with data from column
 * `index`
 *
 * @param ctx    Memory context for allocation
 * @param src    Source tensor
 * @param index  Row or column index to extract (0-based)
 * @param axis   Either AxisRow or AxisColumn to specify slice direction
 *
 * @return Float array allocated via arena, or NULL on error
 */
float *tensor_slice(TensorContext *ctx, Tensor *src, size_t index, Axis axis);

/**
 * @brief Creates a new tensor with specified row or column removed.
 *
 * Returns a new tensor with one row or column excluded from the source tensor.
 * The original tensor is unchanged.
 *
 * Result shapes:
 *   - AxisRow: Returns tensor of shape (rows-1, cols) with row `index` removed
 *   - AxisColumn: Returns tensor of shape (rows, cols-1) with column `index`
 * removed
 *
 * @param ctx    Memory context for allocation
 * @param src    Source tensor
 * @param index  Row or column index to remove (0-based)
 * @param axis   Either AxisRow or AxisColumn to specify drop direction
 *
 * @return New tensor with row/column removed, or NULL on error
 */
Tensor *tensor_drop(TensorContext *ctx, Tensor *src, size_t index, Axis axis);

/**
 * @brief Computes the sum of all elements in a tensor.
 *
 * @param tensor  Tensor to sum
 *
 * @return Sum of all elements, or NAN on error
 */
float tensor_sum(Tensor *tensor);

/**
 * @brief Computes the average of all elements in a tensor.
 *
 * @param tensor  Tensor to average
 *
 * @return Average of all elements (sum / (rows * cols)), or NAN on error
 */
float tensor_avg(Tensor *tensor);

/**
 * @brief Duplicates a tensor.
 *
 * Creates a new tensor with the same dimensions and data as the source tensor.
 * The new tensor is allocated using the same memory context as the source.
 *
 * @param ctx  Memory context for allocation
 * @param src  Source tensor
 *
 * @return Pointer to new tensor, or NULL on error
 */
Tensor *tensor_dup(TensorContext *ctx, Tensor *src);

/**
 * @brief Creates a new tensor initialized with zeros.
 *
 * Allocates a new tensor with the specified dimensions and fills it with zero
 * values. All elements are initialized to 0.0f.
 *
 * @param ctx   Memory context for allocation
 * @param rows  Number of rows in the tensor
 * @param cols  Number of columns in the tensor
 *
 * @return Pointer to new zero-filled tensor, or NULL on allocation failure
 */
Tensor *tensor_zeros(TensorContext *ctx, size_t rows, size_t cols);

#endif // CTORCH_TENSOR_H
