#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"

// hls-fpga-machine-learning insert layer-config
struct config2 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 64;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 1024;
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef layer2_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef relu1_table_t table_t;
};

struct config5 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 32;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 2048;
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef layer5_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef relu2_table_t table_t;
};

struct config8 : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 32;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 1024;
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef layer8_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config10 : nnet::activ_config {
    static const unsigned n_in = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef relu3_table_t table_t;
};

struct config11 : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 5;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 160;
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef layer11_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config13 : nnet::activ_config {
    static const unsigned n_in = 5;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef softmax_exp_table_t exp_table_t;
    typedef softmax_inv_table_t inv_table_t;
};


#endif
