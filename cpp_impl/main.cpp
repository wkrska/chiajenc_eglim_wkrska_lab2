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
#include <sys/time.h>

int main(int argc, char *argv[]) {
  struct timeval start_time, end_time;
  bool mismatch[2] = { false, false };

  cnndata_t *ptr_input, *ptr_weight[2], *ptr_output[2];
  cnndata_t *ref_input, *ref_weight[2], *ref_output[2];

  // Compute the size of array in bytes
  uint64_t num_elem_inputs = BATCH_SIZE * N_IFM(0) * R_IFM(0) * C_IFM(0);
  uint64_t num_elem_weights[2] = { M_OFM(0) * N_IFM(0) * K_WTS * K_WTS,
                                   M_OFM(1) * N_IFM(1) * K_WTS * K_WTS };
  uint64_t num_elem_outputs[2] = { BATCH_SIZE * M_OFM(0) * R_OFM(0) * C_OFM(0),
                                   BATCH_SIZE * M_OFM(1) * R_OFM(1) *
                                       C_OFM(1) };

#ifdef __VITIS_CL__
  // Hard coding xclbin filenames, ignoring command line arguments
  std::string xclbinFilename[1] = { "binary_container_1.xclbin" };
#endif

  print_params(0);
  print_params(1);

  cl_object cl_obj;

  krnl_object cnn_obj;
  cnn_obj.index = 0;
  cnn_obj.name = "krnl_cnn_layerX";

#ifdef __VITIS_CL__
  std::cout << "===== Initialize device ======" << std::endl;
  initialize_device(cl_obj);

  std::cout << "===== Reading xclbin ======" << std::endl;
  // Read cnn
  read_xclbin(xclbinFilename[0], cl_obj.bins);

  std::cout << "\n===== Programming kernel ======" << std::endl;
  program_kernel(cl_obj, cnn_obj);
#endif

  std::cout << "\n===== Allocating buffers ======" << std::endl;
  uint64_t bufIdx = 0;
  allocate_readonly_mem(cl_obj, (void **)&ptr_input, bufIdx++,
                        num_elem_inputs * sizeof(cnndata_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_weight[0], bufIdx++,
                        num_elem_weights[0] * sizeof(cnndata_t));
  allocate_readwrite_mem(cl_obj, (void **)&ptr_output[0], bufIdx++,
                         num_elem_outputs[0] * sizeof(cnndata_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_weight[1], bufIdx++,
                        num_elem_weights[1] * sizeof(cnndata_t));
  allocate_readwrite_mem(cl_obj, (void **)&ptr_output[1], bufIdx++,
                         num_elem_outputs[1] * sizeof(cnndata_t));

  MALLOC_CHECK(ref_input = new cnndata_t[num_elem_inputs]);
  MALLOC_CHECK(ref_weight[0] = new cnndata_t[num_elem_weights[0]]);
  MALLOC_CHECK(ref_output[0] = new cnndata_t[num_elem_outputs[0]]);
  MALLOC_CHECK(ref_weight[1] = new cnndata_t[num_elem_weights[1]]);
  MALLOC_CHECK(ref_output[1] = new cnndata_t[num_elem_outputs[1]]);

  // Set randomized inputs in reference copy
  initialize_buffer(ref_input, num_elem_inputs, true);
  initialize_buffer(ref_weight[0], num_elem_weights[0], true);
  initialize_buffer(ref_weight[1], num_elem_weights[1], true);

  // copy ref copy to kernel use copy, coverting to kernel expected layout
  COPY_BUF4D(ref_input, ARRAY4, ptr_input, ARRAYi, BATCH_SIZE, N_IFM(0),
             R_IFM(0), C_IFM(0));
  COPY_BUF4D(ref_weight[0], ARRAY4, ptr_weight[0], ARRAYw, M_OFM(0), N_IFM(0),
             K_WTS, K_WTS);
  COPY_BUF4D(ref_weight[1], ARRAY4, ptr_weight[1], ARRAYw, M_OFM(1), N_IFM(1),
             K_WTS, K_WTS);

  // Random initialize output for kernel use
  initialize_buffer(ptr_output[0], num_elem_outputs[0],
                    true); // cannot assume 0'ed
  initialize_buffer(ptr_output[1], num_elem_outputs[1],
                    true); // cannot assume 0'ed

  std::cout << "\n===== Execution and Timing starts ======" << std::endl;
  gettimeofday(&start_time, NULL);

#ifdef __VITIS_CL__
  cnn_run_kernel(cl_obj, cnn_obj);
#else
  krnl_cnn_layerX(ptr_input, ptr_weight[0], ptr_output[0], BATCH_SIZE, R_OFM(0),
                  C_OFM(0), M_OFM(0), N_IFM(0));
  krnl_cnn_layerX(ptr_output[0], ptr_weight[1], ptr_output[1], BATCH_SIZE,
                  R_OFM(1), C_OFM(1), M_OFM(1), N_IFM(1));
#endif

  gettimeofday(&end_time, NULL);
  std::cout << "Execution and Timing finished!\n" << std::endl;

  std::cout << "===== Verification starts ======" << std::endl;
  mismatch[0] = cnn_check(ptr_input, ptr_weight[0], ptr_output[0], ref_input,
                          ref_weight[0], ref_output[0], 0);
  std::cout << "CNN layer 0 TEST " << (mismatch[0] ? "FAILED" : "PASSED")
            << "\n" << std::endl;
  mismatch[1] = cnn_check(ptr_output[0], ptr_weight[1], ptr_output[1],
                          ref_output[0], ref_weight[1], ref_output[1], 1);
  std::cout << "CNN layer 1 TEST " << (mismatch[1] ? "FAILED" : "PASSED")
            << "\n" << std::endl;

  delete[] ref_input;
  delete[] ref_weight[0];
  delete[] ref_output[0];
  delete[] ref_weight[1];
  delete[] ref_output[1];

  deallocate_mem(cl_obj, ptr_input, 0);
  deallocate_mem(cl_obj, ptr_weight[0], 1);
  deallocate_mem(cl_obj, ptr_output[0], 2);
  deallocate_mem(cl_obj, ptr_weight[1], 3);
  deallocate_mem(cl_obj, ptr_output[1], 4);

  std::cout << "===== Reporting measured throughput ======" << std::endl;
  float timeusec = (end_time.tv_sec - start_time.tv_sec) * 1e6 +
                   (end_time.tv_usec - start_time.tv_usec);
  printf("Runtime = %0.1f (microsec) \n\n", timeusec);
  double num_operations[2] = { BATCH_SIZE * (double)2.0 * M_OFM(0) * R_OFM(0) *
                                   C_OFM(0) * N_IFM(0) * K_WTS * K_WTS,
                               BATCH_SIZE * (double)2.0 * M_OFM(1) * R_OFM(1) *
                                   C_OFM(1) * N_IFM(1) * K_WTS * K_WTS };
  printf("# of operations = %.0f + %.0f \n", num_operations[0],
         num_operations[1]);
  printf("Throughput: %.5f GigaOP/sec\n",
         (double)1.0e-3 * (num_operations[0] + num_operations[1]) / timeusec);

  std::cout << "\n===== Exiting ======" << std::endl;
  return (mismatch[0] || mismatch[1]);
}
