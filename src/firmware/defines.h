#ifndef DEFINES_H_
#define DEFINES_H_

/*
 * Intel HLS makes use of three streaming interfaces:
 *   (1) stream_in - used as the main input to a component
 *   (2) stream_out - used as the main output of a component
 *   (3) stream - allows both reading and writing; used for inter-component connections
 * ihc::stream has a implicitly deleted constructor and therefore, cannot be used as the output of a function/component
 * Therefore, variables of type 'stream' are always passed by reference
 */


#include <sycl/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed_math.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// to ignore this description in the HLS code
#define hls_register

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_LAYER_2 64
#define N_LAYER_2 64
#define N_LAYER_5 32
#define N_LAYER_5 32
#define N_LAYER_8 32
#define N_LAYER_8 32
#define N_LAYER_11 5
#define N_RESULT 5

// hls-fpga-machine-learning insert layer-precision
typedef ac_fixed<16,6,true> input_t;
typedef ac_fixed<16,6,true> model_default_t;
typedef ac_fixed<16,6,true> layer2_t;
typedef ac_int<1, false> layer2_index;
typedef ac_fixed<16,6,true> layer4_t;
typedef ac_fixed<18,8,true> relu1_table_t;
typedef ac_fixed<16,6,true> layer5_t;
typedef ac_int<1, false> layer5_index;
typedef ac_fixed<16,6,true> layer7_t;
typedef ac_fixed<18,8,true> relu2_table_t;
typedef ac_fixed<16,6,true> layer8_t;
typedef ac_int<1, false> layer8_index;
typedef ac_fixed<16,6,true> layer10_t;
typedef ac_fixed<18,8,true> relu3_table_t;
typedef ac_fixed<16,6,true> layer11_t;
typedef ac_int<1, false> layer11_index;
typedef ac_fixed<16,6,true> result_t;
typedef ac_fixed<18,8,true> softmax_table_t;
typedef ac_fixed<18,8,true,AC_RND,AC_SAT> softmax_exp_table_t;
typedef ac_fixed<18,8,true,AC_RND,AC_SAT> softmax_inv_table_t;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
