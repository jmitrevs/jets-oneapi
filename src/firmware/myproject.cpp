#include "myproject.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w11.h"
#include "weights/b11.h"

streaming_pipelined_interface void MyProject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    auto inputsArr = InPipe::read();
    auto inputsCArr = inputsArr.data();

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2] hls_register;
    nnet::dense_resource<input_t, layer2_t, config2>(inputsCArr, layer2_out, w2, b2);

    layer4_t layer4_out[N_LAYER_2] hls_register;
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out);

    layer5_t layer5_out[N_LAYER_5] hls_register;
    nnet::dense_resource<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5);

    layer7_t layer7_out[N_LAYER_5] hls_register;
    nnet::relu<layer5_t, layer7_t, relu_config7>(layer5_out, layer7_out);

    layer8_t layer8_out[N_LAYER_8] hls_register;
    nnet::dense_resource<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8);

    layer10_t layer10_out[N_LAYER_8] hls_register;
    nnet::relu<layer8_t, layer10_t, relu_config10>(layer8_out, layer10_out);

    layer11_t layer11_out[N_LAYER_11] hls_register;
    nnet::dense_resource<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11);

    result_t layer13_out[N_RESULT] hls_register;
    nnet::softmax<layer11_t, result_t, softmax_config13>(layer11_out, layer13_out);

    output_data_t outData;

    #pragma unroll
    for (int i = 0; i < N_RESULT; i++) {
        outData[i] = layer13_out[i];
    }
    OutPipe::write(outData);
}


