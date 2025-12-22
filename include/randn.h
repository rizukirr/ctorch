#ifndef CTORCH_RANDN_H
#define CTORCH_RANDN_H

float randn(void);

int random_between(int min, int max);

int random_to(int max);

#ifdef RAND_IMPLEMENTATION

#include <math.h>
#include <stdlib.h>

float randn(void) {
  double u1 = (rand() + 1.0) / ((double)RAND_MAX + 1.0);
  double u2 = (rand() + 1.0) / ((double)RAND_MAX + 1.0);
  float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

  return z0;
}

int random_between(int min, int max) {
  return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

int random_to(int max) { return random_between(0, max); }

#endif // RAND_IMPLEMENTATION

#endif // CTORCH_RANDN_H
