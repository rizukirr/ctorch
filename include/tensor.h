#ifndef CTORCH_TENSOR_H
#define CTORCH_TENSOR_H

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#define len(src) (sizeof(src) / sizeof(src[0]))

/**
 * @brief 2D matrix/tensor data structure.
 *
 * Represents a dynamically sized 2D array stored in row-major order.
 * Used as the primary tensor type for neural network operations.
 */
typedef struct {
  size_t rows;     // Number of rows
  size_t cols;     // Number of columns
  size_t capacity; // Allocated capacity for growth
  float *grad;     // Gradient tensor (same shape as data)
  float *data;     // Tensor data (rows × cols)
} Tensor;

typedef enum { AxisRow, AxisColum } Axis;

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
 * @param requires_grad If true, enables gradient tracking for tensors created
 *                      with this context (required for backpropagation)
 * @return Pointer to new TensorContext, or NULL on allocation failure
 */
TensorContext *tensor_create(bool requires_grad);

/**
 * @brief Creates a new tensor with specified column count.
 *
 * Allocates a new empty tensor using the arena allocator. The tensor starts
 * with 0 rows and will grow dynamically as data is appended.
 *
 * @param ctx Memory context for allocation
 * @param column_count Number of columns in the tensor
 * @return Pointer to new Tensor, or NULL on allocation failure
 */
Tensor *tensor_new(TensorContext *ctx, const int column_count);

/**
 * @brief Creates a temporary tensor with manual memory management.
 *
 * Creates a tensor using standard malloc/free instead of arena allocation.
 * Must be freed with tensor_free_tmp(). Used for short-lived intermediate
 * results.
 *
 * @param column_count Number of columns in the tensor
 * @return Pointer to new Tensor, or NULL on allocation failure
 */
Tensor *tensor_new_tmp(const int column_count);

/**
 * @brief Prints tensor contents to stdout.
 *
 * Displays the tensor in matrix notation with square brackets.
 * Can optionally shuffle the displayed values for sampling large tensors.
 *
 * @param src Tensor to print
 * @param count Maximum number of rows to print (0 or negative for all rows)
 * @param shuffle If true, randomly samples elements instead of sequential
 * access
 * @return 0 on success, negative error code on failure
 * @note On error, sets global error context via ctorch_set_error()
 */
int tensor_print(Tensor *src, int count, bool shuffle);

/**
 * @brief Frees tensor context and all associated tensors.
 *
 * Releases all memory allocated through the arena allocator, including
 * all tensors created with this context.
 *
 * @param src TensorContext to free
 * @return 0 on success, negative error code on failure
 */
int tensor_free(TensorContext *src);

/**
 * @brief Frees a temporary tensor.
 *
 * Releases memory for a tensor created with tensor_new_tmp().
 * Do not use this for tensors created with tensor_new().
 *
 * @param v Tensor to free
 * @return 0 on success, negative error code on failure
 */
int tensor_free_tmp(Tensor *v);

/**
 * @brief Appends a row to the tensor.
 *
 * Adds a new row to the end of the tensor, automatically growing capacity
 * if needed (doubles capacity when full). Uses arena allocation.
 *
 * @param ctx Memory context for allocation
 * @param dest Tensor to append to
 * @param row_data Array of floats with length equal to tensor's column count
 * @return 0 on success, negative error code on failure
 * @note On error, sets global error context via ctorch_set_error()
 */
int tensor_append(TensorContext *ctx, Tensor *dest, const float *row_data);

/**
 * @brief Appends a row to a temporary tensor.
 *
 * Like tensor_append() but for temporary tensors created with
 * tensor_new_tmp(). Uses standard realloc for memory management.
 *
 * @param dest Temporary tensor to append to
 * @param row_data Array of floats with length equal to tensor's column count
 * @return 0 on success, negative error code on failure
 * @note On error, sets global error context via ctorch_set_error()
 */
int tensor_append_tmp(Tensor *dest, const float *row_data);

/**
 * @brief Appends multiple rows to the tensor.
 *
 * Batch append operation that adds multiple rows in sequence.
 *
 * @param ctx Memory context for allocation
 * @param dest Tensor to append to
 * @param row_data Array of row pointers, each pointing to an array of floats
 * @param size Number of rows to append
 * @return 0 on success, negative error code on failure
 * @note On error, sets global error context via ctorch_set_error()
 */
int tensor_append_all(TensorContext *ctx, Tensor *dest, const float **row_data,
                      size_t size);

/**
 * @brief Gets element at specified row and column.
 *
 * Accesses matrix element using row-major indexing.
 * No bounds checking in release builds.
 *
 * Math: element = data[row * cols + col]
 *
 * @param src Tensor to access
 * @param row AxisRow index (0-based)
 * @param col AxisColum index (0-based)
 * @return Value at the specified position
 */
float tensor_get(const Tensor *src, size_t row, size_t col);

/**
 * @brief Creates a tensor filled with random normal values.
 *
 * Generates a new tensor with values sampled from a standard normal
 * distribution (mean=0, std=1) using Box-Muller transform.
 *
 * Math: values ~ N(0, 1)
 *
 * @param ctx Memory context for allocation
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to new tensor filled with random values, or NULL on error
 */
Tensor *tensor_randn(TensorContext *ctx, size_t rows, size_t cols);

/**
 * @brief Creates an array filled with random normal values.
 *
 * Allocates a 1D array and fills it with values sampled from a standard
 * normal distribution (mean=0, std=1) using Box-Muller transform.
 * The array is allocated using arena allocation and will be freed when
 * the context is freed.
 *
 * Math: values ~ N(0, 1)
 *
 * @param ctx Memory context for allocation
 * @param size Number of elements in the array
 * @return Pointer to float array filled with random values, or NULL on error
 */
float *scalar_randn(TensorContext *ctx, size_t size);

/**
 * @brief Creates a tensor filled with zeros.
 *
 * Allocates a new tensor and initializes all elements to zero.
 * The tensor is allocated using arena allocation and will be freed
 * when the context is freed.
 *
 * @param ctx Memory context for allocation
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to new tensor filled with zeros, or NULL on error
 */
Tensor *tensor_zeros(TensorContext *ctx, size_t rows, size_t cols);

/**
 * @brief Transposes the tensor in-place.
 *
 * Swaps rows and columns of the matrix. After transpose, a matrix of
 * shape (m, n) becomes (n, m).
 *
 * Math: B[j][i] = A[i][j] for all i, j
 *
 * @param src Tensor to transpose (modified in-place)
 * @return 0 on success, negative error code on failure
 * @note On error, sets global error context via ctorch_set_error()
 */
int tensor_transpose(Tensor *src);

/**
 * @brief Selects a single row or column from a tensor.
 *
 * Returns a new tensor containing only the specified row or column from
 * the source tensor. The result is a tensor with either 1 row (if selecting
 * a row) or 1 column (if selecting a column).
 *
 * Result shapes:
 *   - AxisRow: Returns tensor of shape (1, cols) containing the selected row
 *   - AxisColum: Returns tensor of shape (rows, 1) containing the selected
 * column
 *
 * @param ctx Memory context for allocation
 * @param src Source tensor
 * @param index Row or column index to select (0-based)
 * @param axis Either AxisRow or AxisColum to specify selection direction
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
 *   - AxisColum axis: Returns array of length rows with data from column
 * `index`
 *
 * @param ctx Memory context for allocation
 * @param src Source tensor
 * @param index Row or column index to extract (0-based)
 * @param axis Either AxisRow or AxisColum to specify slice direction
 * @return Float array allocated via arena, or NULL on error
 */
float *tensor_slice(TensorContext *ctx, Tensor *src, size_t index, Axis axis);

/**
 * @brief Creates a new tensor with specified row or column removed.
 *
 * Returns a new tensor with one row or column excluded from the source
 * tensor. The original tensor is unchanged.
 *
 * Result shapes:
 *   - AxisRow: Returns tensor of shape (rows-1, cols) with row `index`
 * removed
 *   - AxisColum: Returns tensor of shape (rows, cols-1) with column `index`
 * removed
 *
 * @param ctx Memory context for allocation
 * @param src Source tensor
 * @param index Row or column index to remove (0-based)
 * @param axis Either AxisRow or AxisColum to specify drop direction
 * @return New tensor with row/column removed, or NULL on error
 */
Tensor *tensor_drop(TensorContext *ctx, Tensor *src, size_t index, Axis axis);

/**
 * @brief Computes the sum of each row in a tensor.
 *
 * Returns a new tensor of shape (rows, 1) where each element is the sum
 * of the corresponding row in the source tensor.
 *
 * @param ctx Memory context for allocation
 * @param src Source tensor
 * @return New tensor containing the row sums, or NULL on error
 */
Tensor *tensor_sum_rows(TensorContext *ctx, Tensor *src);

/**
 * @brief Performs element-wise multiplication between two tensors.
 *
 * Computes the element-wise product (Hadamard product) of two tensors.
 * The two tensors must have the same shape.
 *
 * Math formula: Cᵢⱼ = Aᵢⱼ * Bᵢⱼ
 *
 * @param ctx Memory context for allocation
 * @param a The first tensor
 * @param b The second tensor
 * @return Pointer to the output tensor, or NULL on error
 */
Tensor *tensor_mul_elementwise(TensorContext *ctx, Tensor *a, Tensor *b);

/**
 * @brief Creates a deep copy of a tensor.
 *
 * Allocates a new tensor and copies all data from the source tensor.
 * The gradient field is NOT copied (set to NULL in the new tensor).
 *
 * @param ctx Memory context for allocation
 * @param src Source tensor to copy
 * @return Pointer to new tensor copy, or NULL on error
 */
Tensor *tensor_copy(TensorContext *ctx, Tensor *src);

/**
 * @brief Allocates gradient memory for a tensor.
 *
 * Lazily allocates the gradient array for a tensor using arena allocation.
 * If gradient is already allocated, this is a no-op. The gradient array
 * is initialized to zero.
 *
 * @param ctx Memory context for allocation
 * @param t Tensor to allocate gradient for
 * @return 0 on success, negative error code on failure
 */
int tensor_allocate_grad(TensorContext *ctx, Tensor *t);

/**
 * @brief Zeros out the gradient of a tensor.
 *
 * Sets all gradient values to zero. If the gradient array is not allocated,
 * this is a no-op.
 *
 * @param t Tensor whose gradient should be zeroed
 */
void tensor_zero_grad(Tensor *t);

#endif // CTORCH_TENSOR_H
