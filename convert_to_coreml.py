import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.neural_network import datatypes, NeuralNetworkBuilder
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

def create_coreml_model(model_dir, out_dir, mean, std):
    torch_model = torch.load(model_dir)
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

    # Scale the input image
    preprocessing = nn.preprocessing.add()
    preprocessing.scaler.blueBias = -(mean / std)
    preprocessing.scaler.greenBias = -(mean / std)
    preprocessing.scaler.redBias = -(mean / std)
    preprocessing.scaler.channelScale = (1 / 255) * (1 / std)

    # Add argmax layer
    new_layer = nn.layers.add()
    new_layer.name = "argmax"
    params = ct.proto.NeuralNetwork_pb2.ReduceLayerParams
    new_layer.reduce.mode = params.ARGMAX
    new_layer.reduce.axis = params.C

    new_layer.output.append(nn.layers[-2].output[0])
    nn.layers[-2].output[0] = nn.layers[-2].name + "_output"
    new_layer.input.append(nn.layers[-2].output[0])

    # Add squeeze layer
    new_layer = nn.layers.add()
    new_layer.name = "squeeze"
    params = ct.proto.NeuralNetwork_pb2.SqueezeLayerParams
    new_layer.squeeze.squeezeAll = True
    new_layer.output.append(nn.layers[-2].output[0])
    nn.layers[-2].output[0] = nn.layers[-2].name + "_output"
    new_layer.input.append(nn.layers[-2].output[0])

    spec.description.output[0].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32

    output_names = [out.name for out in spec.description.output]
    ct.utils.rename_feature(spec, output_names[0], 'class_prediction')

    output = spec.description.output[0]
    output.type.imageType.colorSpace = ft.ImageFeatureType.GRAYSCALE
    output.type.imageType.height = 512
    output.type.imageType.width = 512

    ct.models.utils.save_spec(spec, out_dir)



def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    parser.add_argument("model_dir", type=str,
                        help='Path to the model to convert')

    parser.add_argument("out_dir", type=str,
                        help='Path to save the model to')

    parser.add_argument("--mean", type=float, default=0.45,
                        help='Mean offset to apply to the data')

    parser.add_argument("--std", type=float, default=0.225,
                        help='Std scaling to apply to the model')


    args = parser.parse_args()

    create_coreml_model(args.model_dir, args.out_dir, args.mean, args.std)


if __name__ == "__main__":
    main()
