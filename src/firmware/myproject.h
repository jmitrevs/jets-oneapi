#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// This file defines the interface to the kernel


using input_data_t = std::array<input_t, N_INPUT_1_1>;
using output_data_t = std::array<result_t, N_RESULT>;

class InPipeID;
class OutPipeID;

using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>));

using InPipe = sycl::ext::intel::experimental::pipe<InPipeID, input_data_t, 0, PipeProps>;
using OutPipe = sycl::ext::intel::experimental::pipe<OutPipeID, output_data_t, 0, PipeProps>;

class MyProjectID;

struct MyProject {
    streaming_pipelined_interface SYCL_EXTERNAL void operator()() const;
};


#endif
