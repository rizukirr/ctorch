#include "errors.h"
#include "arena.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ERROR_ARENA_SIZE 1024 * 4  // 4KB for error messages

struct CtorchErrorContext {
  char *error_msg;
  CTorchError error_code;
  Arena *arena;
};

// Global singleton error context
static CtorchErrorContext *g_error_ctx = NULL;

int ctorch_error_init(void) {
  if (g_error_ctx) {
    return 0; // Already initialized
  }

  Arena *arena = arena_create(ERROR_ARENA_SIZE);
  if (!arena) {
    return -1;
  }

  g_error_ctx = calloc(1, sizeof(CtorchErrorContext));
  if (!g_error_ctx) {
    arena_free(arena);
    return -1;
  }

  g_error_ctx->arena = arena;
  g_error_ctx->error_code = 0;
  g_error_ctx->error_msg = NULL;
  return 0;
}

void ctorch_error_cleanup(void) {
  if (!g_error_ctx) {
    return;
  }

  if (g_error_ctx->arena) {
    arena_free(g_error_ctx->arena);
  }
  free(g_error_ctx);
  g_error_ctx = NULL;
}

CtorchErrorContext *ctorch_error_context(void) {
  if (!g_error_ctx) {
    ctorch_error_init();
  }
  return g_error_ctx;
}

void ctorch_set_error_on(CtorchErrorContext *ctx, CTorchError code, const char *msg) {
  if (!ctx) {
    return;
  }

  ctx->error_code = code;
  ctx->error_msg = NULL;

  // Allocate and copy error message using arena
  if (msg) {
    size_t len = strlen(msg) + 1;
    ctx->error_msg = arena_alloc(ctx->arena, len, ARENA_ALIGNOF(char));
    if (ctx->error_msg) {
      memcpy(ctx->error_msg, msg, len);
    }
  }
}

void ctorch_set_error(CTorchError code, const char *msg) {
  CtorchErrorContext *ctx = ctorch_error_context();
  ctorch_set_error_on(ctx, code, msg);
}

void ctorch_set_error_fmt(CTorchError code, const char *fmt, ...) {
  if (!fmt) {
    return;
  }

  CtorchErrorContext *ctx = ctorch_error_context();
  if (!ctx) {
    return;
  }

  ctx->error_code = code;
  ctx->error_msg = NULL;

  // Use a safe maximum buffer size for error messages
  #define MAX_ERROR_MSG_LEN 512
  char buffer[MAX_ERROR_MSG_LEN];

  va_list args;
  va_start(args, fmt);
  int len = vsnprintf(buffer, MAX_ERROR_MSG_LEN, fmt, args);
  va_end(args);

  // vsnprintf returns the number of characters that would have been written
  if (len < 0) {
    // Encoding error - fall back to simple message
    ctorch_set_error(code, "Error formatting error message");
    return;
  }

  // Allocate and copy formatted message using arena
  size_t actual_len = (len >= MAX_ERROR_MSG_LEN) ? MAX_ERROR_MSG_LEN - 1 : len;
  ctx->error_msg = arena_alloc(ctx->arena, actual_len + 1, ARENA_ALIGNOF(char));
  if (ctx->error_msg) {
    memcpy(ctx->error_msg, buffer, actual_len + 1);
  }
  #undef MAX_ERROR_MSG_LEN
}

char *ctorch_get_error_from(CtorchErrorContext *ctx) {
  if (!ctx) {
    return NULL;
  }
  return ctx->error_msg;
}

char *ctorch_get_error(void) {
  CtorchErrorContext *ctx = ctorch_error_context();
  return ctorch_get_error_from(ctx);
}

CTorchError ctorch_get_error_code(void) {
  CtorchErrorContext *ctx = ctorch_error_context();
  if (!ctx) {
    return 0;
  }
  return ctx->error_code;
}
