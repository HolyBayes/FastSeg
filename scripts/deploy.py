import sys; sys.path.append('../'); sys.path.append('../modules/'); sys.path.append('../models/')
from models.SERNet_Former import SERNet_Former
import torch
import torch.onnx
from onnx import ModelProto
import tensorrt as trt
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import onnxruntime as ort
import time
from tqdm import trange


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


if __name__ == '__main__':
    model = SERNet_Former(n_classes=2)
    model.eval()

    
    # PyTorch -> ONNX

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 1024, 768)  # Adjust the shape as necessary

    onnx_path = "sernet.onnx"
    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_path, 
                    export_params=True, 
                    opset_version=11, 
                    do_constant_folding=True, 
                    input_names=['input'], 
                    output_names=['output'], 
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Print a human-readable representation of the graph
    print(onnx.helper.printable_graph(onnx_model.graph))

    # Example input data
    input_data = np.random.rand(1, 3, 1024, 768).astype(np.float32)
    # Create an inference session
    # session = ort.InferenceSession(onnx_path)
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

    # Get model input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    n_warmup_iters = 100
    print('ONNX model warmup:')
    for _ in range(n_warmup_iters):
        result = session.run([output_name], {input_name: input_data})[0]
    # Run the model (inference)
    print('ONNX model performance evaluation:')
    n_eval_iters = 100
    start_ts = time.time()
    for _ in range(n_eval_iters):
        result = session.run([output_name], {input_name: input_data})[0]
    print(f'Inference speed (ms): {1000*(time.time()-start_ts)/n_eval_iters:.4f}')

    assert result.shape == (1,2,1024,768)


    # # ONNX -> TensorRT

    import mmcv
    from mmcv.tensorrt import onnx2tensorrt

    # Define the path for the TensorRT engine file
    trt_file_path = "sernet_model.trt"

    # Convert ONNX to TensorRT
    onnx2tensorrt(
        onnx_path,
        trt_file_path,
        input_shapes={'input': [1, 3, 1024, 768]},  # Adjust input shapes if necessary
        max_workspace_size=1 << 30,  # 1 GB
        fp16_mode=False  # Set to True if you want to use FP16 precision
    )

    # Met some hardware issues


    # engine = backend.prepare(onnx_model, device='CUDA:0')
    # input_data = np.random.random(size=(1, 3, 1024, 768)).astype(np.float32)
    # output_data = engine.run(input_data)[0]
    # print(output_data)
    # print(output_data.shape)

        
