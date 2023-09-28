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

#pragma once

/*
 * CMU 18643 Fall 2022 Lab Exercise
 *
 * The parameters in this file sets the CNN kernel on FPGA
 *
 * You can edit this file
 */
#include "util643.h"
#include "instance643.h"

typedef uint32_t index_t;

// blocked
#define TR (4) // output row
#define TC (4) // output column
#define TM (4) // output depth
#define TN (4) // input depth

/*
 * Access macros for the input and output buffers.
 * You can change this from standard natural layout
 *
 * Note: if layer 0 and layer 1 use the same kernel, ARRAYi and ARRAYo has to
 *match
 */
#define ARRAYi(ptr, iB, iN, iR, iC, dB, dN, dR, dC)                            \
  ((ptr)[(iB) * (dN) * (dR) * (dC) + (iN) * (dR) * (dC) + (iR) * (dC) + (iC)])

#define ARRAYo(ptr, iB, iM, iR, iC, dB, dM, dR, dC)                            \
  ((ptr)[(iB) * (dM) * (dR) * (dC) + (iM) * (dR) * (dC) + (iR) * (dC) + (iC)])

#define ARRAYw(ptr, iM, iN, iR, iC, dM, dN, dR, dC)                            \
  ((ptr)[(iM) * (dN) * (dR) * (dC) + (iN) * (dR) * (dC) + (iR) * (dC) + (iC)])

#ifdef __VITIS_CL__
extern "C"
#endif
    void
krnl_cnn_layerX(const cnndata_t *input, const cnndata_t *weights,
                cnndata_t *output, uint64_t batch_size, uint64_t r_ofm,
                uint64_t c_ofm, uint64_t m_ofm, uint64_t n_ifm);

void cnn_blocked_kernel(
    cnndata_t BufI[TN][TR * S_WTS + K_WTS - S_WTS][TC * S_WTS + K_WTS - S_WTS],
    cnndata_t BufO[TM][TR][TC], cnndata_t BufW[TM][TN][K_WTS][K_WTS]);

void cnn_blocked_kernel_windowed(
    cnndata_t BufI[TN][TR * S_WTS + K_WTS - S_WTS][TC * S_WTS + K_WTS - S_WTS],
    cnndata_t BufO[TM][TR][TC], cnndata_t BufW[TM][TN][K_WTS][K_WTS]);

