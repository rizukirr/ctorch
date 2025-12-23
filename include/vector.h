#ifndef CTORCH_VECTOR_H
#define CTORCH_VECTOR_H

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#define len(src) (sizeof(src) / sizeof(src[0]))

/**
 * @brief 2D matrix/vector data structure.
 *
 * Represents a dynamically sized 2D array stored in row-major order.
 * Used as the primary tensor type for neural network operations.
 */
typedef struct {
  size_t rows;      /** Number of rows in the matrix */
  size_t cols;      /** Number of columns in the matrix */
  size_t capacity;  /** Allocated capacity for growth */
  float *data;      /** Contiguous array storing matrix data in row-major order */
} Vector;

/**
 * @brief Opaque memory context for vector allocation.
 *
 * Manages memory using an arena allocator for efficient batch allocation
 * and deallocation of vectors.
 */
typedef struct VectorContext VectorContext;

/**
 * @brief Creates a new vector memory context.
 *
 * Initializes an arena-based memory context for managing vector allocations.
 * All vectors created with this context will be freed when the context is freed.
 *
 * @return Pointer to new VectorContext, or NULL on allocation failure
 */
VectorContext *vector_create(void);

/**
 * @brief Creates a new vector with specified column count.
 *
 * Allocates a new empty vector using the arena allocator. The vector starts
 * with 0 rows and will grow dynamically as data is appended.
 *
 * @param ctx Memory context for allocation
 * @param column_count Number of columns in the vector
 * @return Pointer to new Vector, or NULL on allocation failure
 */
Vector *vector_new(VectorContext *ctx, const int column_count);

/**
 * @brief Creates a temporary vector with manual memory management.
 *
 * Creates a vector using standard malloc/free instead of arena allocation.
 * Must be freed with vector_free_tmp(). Used for short-lived intermediate results.
 *
 * @param column_count Number of columns in the vector
 * @return Pointer to new Vector, or NULL on allocation failure
 */
Vector *vector_new_tmp(const int column_count);

/**
 * @brief Prints vector contents to stdout.
 *
 * Displays the vector in matrix notation with square brackets.
 * Can optionally shuffle the displayed values for sampling large vectors.
 *
 * @param src Vector to print
 * @param count Maximum number of rows to print (0 or negative for all rows)
 * @param suffle If true, randomly samples elements instead of sequential access
 */
void vector_print(Vector *src, int count, bool suffle);

/**
 * @brief Frees vector context and all associated vectors.
 *
 * Releases all memory allocated through the arena allocator, including
 * all vectors created with this context.
 *
 * @param src VectorContext to free
 */
void vector_free(VectorContext *src);

/**
 * @brief Frees a temporary vector.
 *
 * Releases memory for a vector created with vector_new_tmp().
 * Do not use this for vectors created with vector_new().
 *
 * @param v Vector to free
 */
void vector_free_tmp(Vector *v);

/**
 * @brief Appends a row to the vector.
 *
 * Adds a new row to the end of the vector, automatically growing capacity
 * if needed (doubles capacity when full). Uses arena allocation.
 *
 * @param ctx Memory context for allocation
 * @param dest Vector to append to
 * @param row_data Array of floats with length equal to vector's column count
 */
void vector_append(VectorContext *ctx, Vector *dest, const float *row_data);

/**
 * @brief Appends a row to a temporary vector.
 *
 * Like vector_append() but for temporary vectors created with vector_new_tmp().
 * Uses standard realloc for memory management.
 *
 * @param dest Temporary vector to append to
 * @param row_data Array of floats with length equal to vector's column count
 */
void vector_append_tmp(Vector *dest, const float *row_data);

/**
 * @brief Appends multiple rows to the vector.
 *
 * Batch append operation that adds multiple rows in sequence.
 *
 * @param ctx Memory context for allocation
 * @param dest Vector to append to
 * @param row_data Array of row pointers, each pointing to an array of floats
 * @param size Number of rows to append
 */
void vector_append_all(VectorContext *ctx, Vector *dest,
                       const float **row_data, size_t size);

/**
 * @brief Gets element at specified row and column.
 *
 * Accesses matrix element using row-major indexing.
 * No bounds checking in release builds.
 *
 * Math: element = data[row * cols + col]
 *
 * @param src Vector to access
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @return Value at the specified position
 */
float vector_get(const Vector *src, size_t row, size_t col);

/**
 * @brief Creates a vector filled with random normal values.
 *
 * Generates a new vector with values sampled from a standard normal
 * distribution (mean=0, std=1) using Box-Muller transform.
 *
 * Math: values ~ N(0, 1)
 *
 * @param ctx Memory context for allocation
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to new vector filled with random values, or NULL on error
 */
Vector *vector_randn(VectorContext *ctx, size_t rows, size_t cols);

/**
 * @brief Transposes the vector in-place.
 *
 * Swaps rows and columns of the matrix. After transpose, a matrix of
 * shape (m, n) becomes (n, m).
 *
 * Math: B[j][i] = A[i][j] for all i, j
 *
 * @param src Vector to transpose (modified in-place)
 */
void vector_transpose(Vector *src);

/**
 * @brief Extracts a single column as a new vector.
 *
 * Creates a new vector containing only the specified column from the source.
 * The result is a column vector with shape (rows, 1).
 *
 * @param ctx Memory context for allocation
 * @param src Source vector
 * @param col Column index to extract (0-based)
 * @return New vector containing the column, or NULL on error
 */
Vector *vector_get_column(VectorContext *ctx, Vector *src, size_t col);

/**
 * @brief Creates a new vector with specified column removed.
 *
 * Returns a new vector containing all columns except the one at the specified index.
 * Original vector is unchanged. Result has shape (rows, cols-1).
 *
 * @param ctx Memory context for allocation
 * @param src Source vector
 * @param col Column index to remove (0-based)
 * @return New vector with column removed, or NULL on error
 */
Vector *vector_remove_column(VectorContext *ctx, Vector *src, size_t col);

#endif // CTORCH_VECTOR_H
