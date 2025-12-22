#ifndef CTORCH_RANDN_H
#define CTORCH_RANDN_H

/**
 * @brief Generates a random number from standard normal distribution.
 *
 * Uses Box-Muller transform to convert uniform random numbers into
 * normally distributed values with mean=0 and standard deviation=1.
 *
 * Math formula:
 *   z = sqrt(-2 * ln(u1)) * cos(2π * u2)
 * where u1, u2 are uniform random numbers in (0, 1)
 *
 * @return Random float from N(0, 1) distribution
 */
float randn(void);

/**
 * @brief Generates a random integer in the specified range (inclusive).
 *
 * Returns a uniformly distributed random integer between min and max,
 * including both endpoints.
 *
 * @param min Minimum value (inclusive)
 * @param max Maximum value (inclusive)
 * @return Random integer in range [min, max]
 */
int random_between(int min, int max);

/**
 * @brief Generates a random integer from 0 to max (inclusive).
 *
 * Convenience function equivalent to random_between(0, max).
 *
 * @param max Maximum value (inclusive)
 * @return Random integer in range [0, max]
 */
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
