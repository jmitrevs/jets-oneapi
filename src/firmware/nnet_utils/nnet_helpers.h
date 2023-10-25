#ifndef NNET_HELPERS_H
#define NNET_HELPERS_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

namespace nnet {

template <class srcType, class dstType, size_t SIZE> void convert_data(srcType *src, dstType *dst) {
    for (size_t i = 0; i < SIZE; i++) {
        dst[i] = dstType(src[i]);
    }
}

template <class srcType, class dstType, size_t SIZE> void convert_data_back(srcType *src, dstType *dst) {
    for (size_t i = 0; i < SIZE; i++) {
        dst[i] = static_cast<dstType>(src[i].to_double());
    }
}

extern bool trace_enabled;
extern std::map<std::string, void *> *trace_outputs;
extern size_t trace_type_size;

constexpr int ceillog2(int x) { return (x <= 2) ? 1 : 1 + ceillog2((x + 1) / 2); }

constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }

constexpr int pow2(int x) { return x == 0 ? 1 : 2 * pow2(x - 1); }

template <class data_T, class save_T> void save_output_array(data_T *data, save_T *ptr, size_t layer_size) {
    for (int i = 0; i < layer_size; i++) {
        ptr[i] = static_cast<save_T>(data[i].to_double());
    }
}


// We don't want to include save_T in this function because it will be inserted into myproject.cpp
// so a workaround with element size is used
template <class data_T> void save_layer_output(data_T *data, const char *layer_name, size_t layer_size) {
    if (!trace_enabled)
        return;

    if (trace_outputs) {
        if (trace_outputs->count(layer_name) > 0) {
            if (trace_type_size == 4) {
                save_output_array(data, (float *)(*trace_outputs)[layer_name], layer_size);
            } else if (trace_type_size == 8) {
                save_output_array(data, (double *)(*trace_outputs)[layer_name], layer_size);
            } else {
                std::cout << "Unknown trace type!" << std::endl;
            }
        } else {
            std::cout << "Layer name: " << layer_name << " not found in debug storage!" << std::endl;
        }
    } else {
        std::ostringstream filename;
        filename << "./tb_data/" << layer_name << "_output.log"; // TODO if run as a shared lib, path should be ../tb_data
        std::fstream out;
        out.open(filename.str(), std::ios::app);
        assert(out.is_open());
        for (int i = 0; i < layer_size; i++) {
            out << data[i] << " "; // We don't care about precision in text files
        }
        out << std::endl;
        out.close();
    }
}

} // namespace nnet

#endif
