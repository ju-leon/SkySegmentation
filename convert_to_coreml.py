import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft
import torch
import argparse

def get_nn(spec):
    if spec.WhichOneof("Type") == "neuralNetwork":
        return spec.neuralNetwork
    elif spec.WhichOneof("Type") == "neuralNetworkClassifier":
        return spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
        return spec.neuralNetworkRegressor
    else:
        raise ValueError("MLModel does not have a neural network")


def convert_multiarray_to_image(feature, is_bgr=False):
    import coremltools.proto.FeatureTypes_pb2 as ft

    if feature.type.WhichOneof("Type") != "multiArrayType":
        raise ValueError("%s is not a multiarray type" % feature.name)

    shape = tuple(feature.type.multiArrayType.shape)
    channels = None
    if len(shape) == 2:
        channels = 1
        height, width = shape
    elif len(shape) == 3:
        channels, height, width = shape

    if channels != 1 and channels != 3:
        raise ValueError("Shape {} not supported for image type".format(shape))

    if channels == 1:
        feature.type.imageType.colorSpace = ft.ImageFeatureType.GRAYSCALE
    elif channels == 3:
        if is_bgr:
            feature.type.imageType.colorSpace = ft.ImageFeatureType.BGR
        else:
            feature.type.imageType.colorSpace = ft.ImageFeatureType.RGB

    feature.type.imageType.width = width
    feature.type.imageType.height = height


def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    parser.add_argument("model_dir", type=str,
                        help='Path to the model to convert')

    parser.add_argument("out_dir", type=str,
                        help='Path to save the model to')

    args = parser.parse_args()
    
    torch_model = torch.load(args.model_dir)
    torch_model.eval()

    # Trace model
    example_input = torch.rand(1, 3, 512, 512)
    traced_model = torch.jit.trace(torch_model, example_input)
    out = traced_model(example_input)

    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)]
    )

    model.author = "Leon Jungemeyer"
    model.version = '1.0'
    model.short_description = "Segementation model to seperate sky and foreground."

    spec = model.get_spec()
    input_names = [inp.name for inp in spec.description.input]
    ct.utils.rename_feature(spec, input_names[0], 'image')

    input = spec.description.input[0]
    input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    input.type.imageType.height = 512
    input.type.imageType.width = 512

    # Add argmax layer to the end of the model to only select highest probability class
    nn = get_nn(spec)
    new_layer = nn.layers.add()
    new_layer.name = "argmax"
    params = ct.proto.NeuralNetwork_pb2.ReduceLayerParams
    new_layer.reduce.mode = params.ARGMAX
    new_layer.reduce.axis = params.C

    # Change input and output type to int
    new_layer.output.append(nn.layers[-2].output[0])
    nn.layers[-2].output[0] = nn.layers[-2].name + "_output"
    new_layer.input.append(nn.layers[-2].output[0])

    spec.description.output[0].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32

    output_names = [out.name for out in spec.description.output]
    ct.utils.rename_feature(spec, output_names[0], 'class_prediction')

    ct.models.utils.save_spec(spec, args.out_dir)

if __name__ == "__main__":
    main()
