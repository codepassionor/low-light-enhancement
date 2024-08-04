from unet_structure import UNet

# Function to get upsample parameters
def get_upsample_64x64_params(unet_model):
    # Specify the layers based on the actual model structure
    layer_names = [unet_model.C9]  # Make sure C9 is the correct attribute name
    # Initialize an empty list to store parameters
    params = []
    # Iterate over each specified layer
    for layer in layer_names:
        # Add the parameters of the layer to the list
        params.extend(layer.parameters())
    return params

# Main script execution
if __name__ == '__main__':
    # Model initialization
    unet_model = UNet()
    
    # Get upsample parameters
    upsample_params = get_upsample_64x64_params(unet_model)
    # Print the upsample parameters
    print(upsample_params)
