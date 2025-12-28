#ifndef CTORCH_ERRORS_H
#define CTORCH_ERRORS_H

/**
 * @brief Error codes for ctorch library operations.
 *
 * All error codes are negative values to distinguish them from successful
 * return values (which are typically 0 or positive). These codes are used
 * throughout the library to indicate specific failure conditions.
 *
 * Error codes fall into the following categories:
 * - Memory errors (-1 to -2): Allocation and memory management failures
 * - Parameter errors (-3 to -5): Invalid parameters or validation failures
 * - Index errors (-6 to -7): Out of bounds or invalid indices
 * - Data errors (-8 to -9): NULL or mismatched data
 * - Operation errors (-10 to -11): Failed tensor operations
 */
typedef enum {
  // Memory errors
  CTORCH_ERROR_OUT_OF_MEMORY = -1, ///< malloc/calloc/realloc failed
  CTORCH_ERROR_ARENA_ALLOCATION_FAILED =
      -2, ///< Arena allocator ran out of memory

  // Parameter validation errors
  CTORCH_ERROR_NULL_PARAMETER = -3, ///< NULL pointer passed as parameter
  CTORCH_ERROR_INVALID_SHAPE =
      -4, ///< Invalid tensor dimensions (zero or negative)
  CTORCH_ERROR_DIMENSION_MISMATCH =
      -5, ///< Incompatible tensor dimensions for operation

  // Index/bounds errors
  CTORCH_ERROR_OUT_OF_BOUNDS = -6, ///< Array index exceeds valid range
  CTORCH_ERROR_INVALID_INDEX = -7, ///< Invalid label or class index

  // Data errors
  CTORCH_ERROR_NULL_DATA = -8, ///< Tensor data pointer is NULL
  CTORCH_ERROR_LABEL_MISMATCH =
      -9, ///< Label count doesn't match number of samples

  // Tensor operation errors
  CTORCH_ERROR_INVALID_SLICE = -10,    ///< Invalid slice parameters provided
  CTORCH_ERROR_TRANSPOSE_FAILED = -11, ///< Transpose operation failed
} CTorchError;

/**
 * @brief Opaque error context structure.
 *
 * This structure holds error state including error code and message.
 * Memory for error messages is managed by an internal arena allocator.
 */
typedef struct CtorchErrorContext CtorchErrorContext;

/**
 * @brief Initializes the global error context.
 *
 * Must be called before using any error handling functions.
 * Creates the global error context with its own arena allocator.
 *
 * @return 0 on success, -1 on failure
 */
int ctorch_error_init(void);

/**
 * @brief Cleans up the global error context.
 *
 * Frees all memory associated with the global error context.
 * Should be called at program shutdown.
 */
void ctorch_error_cleanup(void);

/**
 * @brief Gets the global error context.
 *
 * Returns the singleton error context. The context is automatically
 * initialized on first access if not already initialized.
 *
 * @return Pointer to global error context
 */
CtorchErrorContext *ctorch_error_context(void);

/**
 * @brief Sets an error on the global error context.
 *
 * Stores an error code and message in the global context for later retrieval.
 * The message string is copied into the arena allocator.
 *
 * @param code Error code from CTorchError enum
 * @param msg Error message string (will be copied)
 */
void ctorch_set_error(CTorchError code, const char *msg);

/**
 * @brief Sets a formatted error message on the global error context.
 *
 * Similar to ctorch_set_error() but allows printf-style formatting.
 * The formatted message is allocated using the arena allocator.
 *
 * @param code Error code from CTorchError enum
 * @param fmt Printf-style format string
 * @param ... Variable arguments for format string
 */
void ctorch_set_error_fmt(CTorchError code, const char *fmt, ...);

/**
 * @brief Retrieves the last error message from the global error context.
 *
 * Returns the error message string that was previously set.
 * The returned string is owned by the context and should not be freed.
 *
 * @return Error message string, or NULL if no error has been set
 */
char *ctorch_get_error(void);

/**
 * @brief Gets the last error code from the global error context.
 *
 * @return Error code, or 0 if no error has been set
 */
CTorchError ctorch_get_error_code(void);

/**
 * @brief Sets an error on the specified error context.
 *
 * @param ctx Error context
 * @param code Error code from CTorchError enum
 * @param msg Error message string (will be copied)
 */
void ctorch_set_error_on(CtorchErrorContext *ctx, CTorchError code,
                         const char *msg);

/**
 * @brief Gets the error message from the specified error context.
 *
 * @param ctx Error context
 * @return Error message string, or NULL if no error
 */
char *ctorch_get_error_from(CtorchErrorContext *ctx);

#endif // CTORCH_ERRORS_H
