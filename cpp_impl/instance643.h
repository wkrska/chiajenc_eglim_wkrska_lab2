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
 * The parameters in this file sets the problem sizes
 *
 * You cannot change this file in your final submission
 */
#pragma once

typedef int cnndata_t;

static inline bool nearlyEqual(cnndata_t a, cnndata_t b) { return a == b; }

#define K_WTS (4) // weight width and height (square)
                  // same depth as output
#define S_WTS (1) // sliding stride

#if 1 // fullsize

#define BATCH_SIZE 10

//
// Layer 1
//
#define r_ofm1 (64)                            // height
#define c_ofm1 (64)                            // width
#define m_ofm1 (16)                            // depth
#define r_ifm1 (r_ofm1 *S_WTS + K_WTS - S_WTS) // derived height
#define c_ifm1 (c_ofm1 *S_WTS + K_WTS - S_WTS) // derived width
#define n_ifm1 (64)                            // depth

//
// Layer 0
//
#define r_ofm0 (r_ifm1)                        // height
#define c_ofm0 (c_ifm1)                        // width
#define m_ofm0 (n_ifm1)                        // depth
#define r_ifm0 (r_ofm0 *S_WTS + K_WTS - S_WTS) // derived height
#define c_ifm0 (c_ofm0 *S_WTS + K_WTS - S_WTS) // derived width
#define n_ifm0 (4)                             // depth

#else // SW emulation debug small

#define BATCH_SIZE 3

//
// Layer 1
//
#define r_ofm1 (10)                            // height
#define c_ofm1 (11)                            // width
#define m_ofm1 (3)                             // depth
#define r_ifm1 (r_ofm1 *S_WTS + K_WTS - S_WTS) // derived height
#define c_ifm1 (c_ofm1 *S_WTS + K_WTS - S_WTS) // derived width
#define n_ifm1 (4)                             // depth

//
// Layer 0
//
#define r_ofm0 (r_ifm1)                        // height
#define c_ofm0 (c_ifm1)                        // width
#define m_ofm0 (n_ifm1)                        // depth
#define r_ifm0 (r_ofm0 *S_WTS + K_WTS - S_WTS) // derived height
#define c_ifm0 (c_ofm0 *S_WTS + K_WTS - S_WTS) // derived width
#define n_ifm0 (5)                             // depth

#endif

#define R_OFM(layer) ((layer == 1) ? r_ofm1 : r_ofm0)
#define C_OFM(layer) ((layer == 1) ? c_ofm1 : c_ofm0)
#define M_OFM(layer) ((layer == 1) ? m_ofm1 : m_ofm0)
#define R_IFM(layer) ((layer == 1) ? r_ifm1 : r_ifm0)
#define C_IFM(layer) ((layer == 1) ? c_ifm1 : c_ifm0)
#define N_IFM(layer) ((layer == 1) ? n_ifm1 : n_ifm0)
