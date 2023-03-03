import mxnet
import mxnet.onnx.mx2onnx
import numpy as np
import onnx
import onnx.shape_inference
import onnxruntime.tools.symbolic_shape_infer
import onnxsim


def convert_onnx(sym: str, params: str, in_shapes: list[tuple], in_types: np.dtype,
                 output_path: str, dynamic=False, simplify=True):
    # Define dynamic_axes
    if dynamic:
        dynamic_input_shapes = [tuple(['N'] + list(in_shapes[0][1:]))]
    else:
        dynamic_input_shapes = None

    # Export model into ONNX format
    mxnet.onnx.mx2onnx.export_model(sym, params, in_shapes, in_types, output_path, False, dynamic, dynamic_input_shapes)

    # Check exported onnx model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    try:
        onnx_model = onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes(onnx_model)
        onnx.save(onnx_model, output_path)
    except Exception as e:
        print(f'ERROR: {e}, skip symbolic shape inference.')
    onnx.shape_inference.infer_shapes_path(output_path, output_path, check_type=True, strict_mode=True, data_prop=True)

    # Simplify ONNX model
    if simplify:
        model = onnx.load(output_path)
        input_shapes = {model.graph.input[0].name: in_shapes}
        model, check = onnxsim.simplify(model, test_input_shapes=input_shapes)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')


if __name__ == '__main__':
    sym = 'mxnet_files/mnet_cov2-symbol.json'
    params = 'mxnet_files/mnet_cov2-0000.params'
    in_shapes = [(1, 3, 640, 640)]
    in_types = np.float32
    output_path = 'onnx_files/mnet_cov2-0000.onnx'

    convert_onnx(sym, params, in_shapes, in_types, output_path)
