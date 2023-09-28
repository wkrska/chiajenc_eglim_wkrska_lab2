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

#include "krnl_cnn.h"
#include "cnn_helper.h"
#include <string.h>

// Set kernel arguments and execute it
#ifdef __VITIS_CL__
void cnn_run_kernel(cl_object &cl_obj, krnl_object &krnl_obj) {
  cl_int err;

  //
  // Layer 0
  //

  std::cout << "Running kernel for layer 0..." << std::endl;

  // Get i/o buffers from kernel object
  cl::Buffer *buffer_in = &cl_obj.buffers[0];
  cl::Buffer *buffer_wts = &cl_obj.buffers[1];
  cl::Buffer *buffer_out = &cl_obj.buffers[2];

  // Set the kernel Arguments
  uint64_t narg = 0;
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_in));  // input
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_wts)); // weights
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_out)); // output
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++,
                                           (uint64_t)BATCH_SIZE)); // batch size
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, (uint64_t)R_OFM(0))); // Rows
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, (uint64_t)C_OFM(0))); // Cols
  OCL_CHECK(err, err = cl_obj.krnl->setArg(
                     narg++, (uint64_t)M_OFM(0))); // Output channels
  OCL_CHECK(err, err = cl_obj.krnl->setArg(
                     narg++, (uint64_t)N_IFM(0))); // Input channels

  // Data will be migrated to kernel space
  OCL_CHECK(err, err = cl_obj.q.enqueueMigrateMemObjects(
                     { *buffer_in, *buffer_wts }, 0 /* 0 means from host*/));

  // Launch the Kernel; this is nonblocking.
  OCL_CHECK(err, err = cl_obj.q.enqueueTask(*cl_obj.krnl));

  // The result of the previous kernel execution will need to be retrieved in
  // order to view the results. This call will transfer the data from FPGA to
  // source_results vector
  OCL_CHECK(err, cl_obj.q.enqueueMigrateMemObjects({ *buffer_out },
                                                   CL_MIGRATE_MEM_OBJECT_HOST));

  // Wait for all tasks to finish
  OCL_CHECK(err, cl_obj.q.finish());

  //
  // Layer 1
  //

  std::cout << "Running kernel for layer 1..." << std::endl;

  // Get i/o buffers from kernel object
  buffer_in = &cl_obj.buffers[2];
  buffer_wts = &cl_obj.buffers[3];
  buffer_out = &cl_obj.buffers[4];

  // Set the kernel Arguments
  narg = 0;
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_in));  // input
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_wts)); // weights
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_out)); // output
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++,
                                           (uint64_t)BATCH_SIZE)); // batch size
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, (uint64_t)R_OFM(1))); // Rows
  OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, (uint64_t)C_OFM(1))); // Cols
  OCL_CHECK(err, err = cl_obj.krnl->setArg(
                     narg++, (uint64_t)M_OFM(1))); // Output channels
  OCL_CHECK(err, err = cl_obj.krnl->setArg(
                     narg++, (uint64_t)N_IFM(1))); // Input channels

  // Data will be migrated to kernel space
  OCL_CHECK(err, err = cl_obj.q.enqueueMigrateMemObjects(
                     { *buffer_in, *buffer_wts }, 0 /* 0 means from host*/));

  // Launch the Kernel; this is nonblocking.
  OCL_CHECK(err, err = cl_obj.q.enqueueTask(*cl_obj.krnl));

  // The result of the previous kernel execution will need to be retrieved in
  // order to view the results. This call will transfer the data from FPGA to
  // source_results vector
  OCL_CHECK(err, cl_obj.q.enqueueMigrateMemObjects({ *buffer_out },
                                                   CL_MIGRATE_MEM_OBJECT_HOST));

  // Wait for all tasks to finish
  OCL_CHECK(err, cl_obj.q.finish());

  std::cout << "Kernel executions completed" << std::endl;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////
// DO NOT MODIFY BELOW THIS LINE
// ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

static const std::string cnn_error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

// Verifiy a single batch of data
bool verify(cnndata_t *ref, cnndata_t *checkit, uint64_t iter, uint64_t layer) {
  uint64_t row, col, to;

  for (to = 0; to < M_OFM(layer); to++) {
    for (row = 0; row < R_OFM(layer); row++) {
      for (col = 0; col < C_OFM(layer); col++) {
        if (!(nearlyEqual((cnndata_t)ARRAYo(checkit, iter, to, row, col,
                                            BATCH_SIZE, M_OFM(layer),
                                            R_OFM(layer), C_OFM(layer)),
                          (cnndata_t)ARRAY4(ref, iter, to, row, col, BATCH_SIZE,
                                            M_OFM(layer), R_OFM(layer),
                                            C_OFM(layer))))) {
          printf("\n***Result does not match reference: layer = %lu, "
                 "row = %lu, col = %lu***\n",
                 to, row, col);
          return 0;
        }
      }
    }
  }
  return 1;
}

bool cnn_check(cnndata_t *ptr_input, cnndata_t *ptr_weight,
               cnndata_t *ptr_output, cnndata_t *ref_input,
               cnndata_t *ref_weight, cnndata_t *ref_output, uint64_t layer) {
  std::cout << "Verifying cnn result..." << std::endl;

  // Verify the result
  uint64_t mismatch = 0;
  uint64_t iter;

  for (iter = 0; iter < BATCH_SIZE; iter++) {
    ZhangIsfpga15_1_fp(ref_input, ref_output, ref_weight, iter, layer);
    if (!verify(ref_output, ptr_output, iter, layer)) {
      mismatch = 1;
      break;
    }
  }
  return mismatch;
}

void print_params(uint64_t layer) {
  std::cout << "===== Printing the CNN parameters Layer " << layer
            << " ======" << std::endl;

  std::cout << "Batch size: " << (uint64_t)BATCH_SIZE << std::endl;

  printf("Layer Parameters: \nK_wts: \t%d\tS_wts:\t%d\nR_ofm:\t%d\tC_ofm:"
         "\t%d\tM_ofm:\t%d\tN_ifm:\t%d\n",
         K_WTS, S_WTS, R_OFM(layer), C_OFM(layer), M_OFM(layer), N_IFM(layer));

  printf("Kernel Parameters: \nTr: \t%d\tTc:\t%d\tTm:\t%d\tTn:\t%d\n\n", TR, TC,
         TM, TN);
}

void initialize_buffer(cnndata_t *ptr, unsigned size, bool notzero) {
  for (unsigned i = 0; i < size; i++) {
    ptr[i] = notzero ? (rand() % VRANGE) : 0;
  }
}

void ZhangIsfpga15_1_fp(cnndata_t *input, cnndata_t *output, cnndata_t *weights,
                        uint64_t iter, uint64_t layer) {
  uint64_t row, col, to, ti;

  for (row = 0; row < R_OFM(layer); row++) {
    for (col = 0; col < C_OFM(layer); col++) {
      for (to = 0; to < M_OFM(layer); to++) {
        ARRAY4(output, iter, to, row, col, BATCH_SIZE, M_OFM(layer),
               R_OFM(layer), C_OFM(layer)) = 0;
      }
    }
  }

  for (row = 0; row < R_OFM(layer); row++) {
    for (col = 0; col < C_OFM(layer); col++) {
      for (to = 0; to < M_OFM(layer); to++) {
        for (ti = 0; ti < N_IFM(layer); ti++) {
          uint64_t i, j;
          for (i = 0; i < K_WTS; i++) {
            for (j = 0; j < K_WTS; j++) {
              ARRAY4(output, iter, to, row, col, BATCH_SIZE, M_OFM(layer),
                     R_OFM(layer), C_OFM(layer)) +=
                  ARRAY4(weights, to, ti, i, j, M_OFM(layer), N_IFM(layer),
                         K_WTS, K_WTS) *
                  ARRAY4(input, iter, ti, S_WTS * row + i, S_WTS * col + j,
                         BATCH_SIZE, N_IFM(layer), R_IFM(layer), C_IFM(layer));
            }
          }
        }
      }
    }
  }
}
