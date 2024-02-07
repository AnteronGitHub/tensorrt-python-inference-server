#!/usr/bin/env python3
import ctypes
import os
from time import time

from cuda import cudart
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
             builder.create_builder_config() as config, \
             trt.OnnxParser(network, TRT_LOGGER) as parser, \
             trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        _, host_mem = cudart.cudaMallocHost(nbytes)
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        _, self._device = cudart.cudaMalloc(nbytes)
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cudart.cudaFree(self.device)
        cudart.cudaFreeHost(self.host.ctypes.data)

def allocate_memory(engine : trt.ICudaEngine):
    inputs = []
    outputs = []
    bindings = []
    _, stream = cudart.cudaStreamCreate()
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        shape = engine.get_tensor_shape(binding)
        size = trt.volume(shape)
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
        bindingMemory = HostDeviceMem(size, dtype)

        bindings.append(int(bindingMemory.device))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            print("Input shape: {}, type: {}".format(shape, dtype))
            inputs.append(bindingMemory)
        else:
            print("Output shape: {}, type: {}".format(shape, dtype))
            outputs.append(bindingMemory)

    return inputs, outputs, bindings, stream

def run_inference(context, inputs, outputs, bindings, stream):
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream) for inp in inputs]

    context.execute_async_v2(bindings=bindings, stream_handle=stream)

    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream) for out in outputs]

    cudart.cudaStreamSynchronize(stream)

    return [out.host for out in outputs]

if __name__ == '__main__':
    with get_engine("/app/densenet121_Opset16.onnx", "/app/densenet121_Opset16.trt") as engine, \
         engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_memory(engine)

        # Generate random image
        noisy_image = np.random.rand(1,3,608,608).astype('float32')
        inputs[0].host = noisy_image

        n_inferences = 1000
        print("Running {} inferences".format(n_inferences))
        for i in range(n_inferences):
            started = time()
            run_inference(context, inputs, outputs, bindings, stream)
            elapset = time() - started
            print("latency: {} ms".format(elapset*1000.0))
