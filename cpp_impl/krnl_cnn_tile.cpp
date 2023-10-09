/****************************************************************
 * Copyright (c) 2020~2022, 18-643 Course Staff, CMU
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
 * CMU 18643 Fall 2022 Lab Exercise
 *
 * You can edit this file
 */

/****************************************************************
 * Blocked convolution layer implementation
 * based on Figure 5:
 *    C. Zhang, et al., "Optimizing FPGA-based Accelerator
 *    Design for Deep Convolutional Neural Networks," FPGA, 2015.
 ****************************************************************/

#include "krnl_cnn.h"

// Prevent aliasing
#undef BATCH_SIZE
#undef R_OFM
#undef C_OFM
#undef R_IFM
#undef C_IFM
#undef M_OFM
#undef N_IFM

#include "util643.h"

void cnn_blocked_kernel(
    cnndata_t BufI[TN][TR * S_WTS + K_WTS - S_WTS][TC * S_WTS + K_WTS - S_WTS],
    cnndata_t BufO[TM][TR][TC], cnndata_t BufW[TM][TN][K_WTS][K_WTS]) {
#pragma HLS ARRAY_RESHAPE dim=2 type=complete variable=BufW
#pragma HLS ARRAY_PARTITION dim=3 factor=4 type=block variable=BufW
#pragma HLS ARRAY_PARTITION dim=4 factor=4 type=block variable=BufW
#pragma HLS ARRAY_RESHAPE dim=2 type=complete variable=BufI
#pragma HLS ARRAY_RESHAPE dim=3 type=complete variable=BufI
  index_t to_b, ti_b, row_b, col_b;

Row:
  for (row_b = 0; row_b < TR; row_b++) {
  Col:
    for (col_b = 0; col_b < TC; col_b++) {
    To:
      for (to_b = 0; to_b < TM; to_b++) {
#pragma HLS PIPELINE
     Ti:
        for (ti_b = 0; ti_b < TN; ti_b++) {
          index_t i, j;
        Krow:
          for (i = 0; i < K_WTS; i++) {
          Kcol:
            for (j = 0; j < K_WTS; j++) {
              BufO[to_b][row_b][col_b] +=
                  BufW[to_b][ti_b][i][j] *
                  BufI[ti_b][S_WTS * row_b + i][S_WTS * col_b + j];
            }
          }
        }
      }
    }
  }
}


static inline void fetchNewColumn(
    cnndata_t window[K_WTS][K_WTS],
    cnndata_t BufI[TN][TR * S_WTS + K_WTS - S_WTS][TC * S_WTS + K_WTS - S_WTS],
    index_t row_b, index_t col_b, index_t ti_b) {
  // shifting the old columns.
  for (index_t col = 0; col < K_WTS - 1; col++) {
    for (index_t row = 0; row < K_WTS; row++) {
      window[row][col] = window[row][col + 1];
    }
  }

  // read one new column
  for (index_t i = 0; i < K_WTS; i++) {
    if (((S_WTS * row_b + i) < (TR * S_WTS + K_WTS - S_WTS)) &&
        ((col_b) < (TC * S_WTS + K_WTS - S_WTS))) {
      window[i][K_WTS - 1] = BufI[ti_b][S_WTS * row_b + i][col_b];
    } else {
      window[i][K_WTS - 1] = 0;
    }
  }
}

void cnn_blocked_kernel_windowed(
    cnndata_t BufI[TN][TR * S_WTS + K_WTS - S_WTS][TC * S_WTS + K_WTS - S_WTS],
    cnndata_t BufO[TM][TR][TC], cnndata_t BufW[TM][TN][K_WTS][K_WTS]) {

  if (S_WTS != 1) {
    // this only works for stride 1.
    return;
  }

// This version of the kernel implements a K_WTS-by-K_WTS sliding
// window of BufI values. The innder loops only needs to read BufI
// K_WTS times for each round of the Krow/Kcol loops with K_WTS^2
// multiply-accumulates.

Row:
  for (index_t row_b = 0; row_b < TR; row_b++) {
  // col_b loop reorder inside
  To:
    for (index_t to_b = 0; to_b < TM; to_b++) {
    Ti:
      for (index_t ti_b = 0; ti_b < TN; ti_b++) {
        cnndata_t window[K_WTS][K_WTS];

      // prime window buffer with first K_WTS-1 columns before
      // entering main loop
      Priming:
        for (index_t col_b = 0; col_b < K_WTS - 1; col_b++) {
          fetchNewColumn(window, BufI, row_b, col_b, ti_b);
        }

      Col:
        for (index_t col_b = 0; col_b < TC; col_b++) {
          fetchNewColumn(window, BufI, row_b, col_b + K_WTS - 1, ti_b);
        Krow:
          for (index_t i = 0; i < K_WTS; i++) {
          Kcol:
            for (index_t j = 0; j < K_WTS; j++) {
              BufO[to_b][row_b][col_b] += BufW[to_b][ti_b][i][j] * window[i][j];
            }
          }
        }
      }
    }
  }
}
