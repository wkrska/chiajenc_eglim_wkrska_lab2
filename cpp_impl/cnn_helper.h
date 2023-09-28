/****************************************************************
 * Copyright (c) 2017~2022, 18-643 Course Staff, CMU
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.

 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of the FreeBSD Project.
 ****************************************************************/

/*
 * CMU 18643 Fall 2022 Lab1 Exercise
 */

#pragma once

#include "utils.h"
#include "instance643.h"

#ifdef __VITIS_CL__
// Set kernel arguments and execute it
void cnn_run_kernel(cl_object &cl_obj, krnl_object &krnl_obj);
#endif

// Verification functions
bool cnn_check(cnndata_t *ptr_a, cnndata_t *ptr_b, cnndata_t *ptr_result,
               cnndata_t *ref_a, cnndata_t *ref_b, cnndata_t *ref_result,
               uint64_t layer);

// printout cnn problem parameters
void print_params(uint64_t layer);

// Initialize memory with random numbers
void initialize_buffer(cnndata_t *ptr, unsigned size, bool notzero);

// Reference CNN code
void ZhangIsfpga15_1_fp(cnndata_t *input, cnndata_t *output, cnndata_t *weights,
                        uint64_t iter, uint64_t layer);

// Copy between different memory layouts
#define COPY_BUF4D(fromPtr, fromAcc, toPtr, toAcc, dB, dN, dR, dC)             \
  for (int b = 0; b < (dB); b++) {                                             \
    for (int n = 0; n < (dN); n++) {                                           \
      for (int r = 0; r < (dR); r++) {                                         \
        for (int c = 0; c < (dC); c++) {                                       \
          toAcc((toPtr), b, n, r, c, (dB), (dN), (dR), (dC)) =                 \
              fromAcc((fromPtr), b, n, r, c, (dB), (dN), (dR), (dC));          \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
