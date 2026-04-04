import blobconverter

blob_path = blobconverter.from_tf(
    # The 'frozen_pb' is the 1st positional argument. 
    # We give it an empty string so it stops complaining.
    "", 
    saved_model="computer_vision/digit_model_saved",
    data_type="FP16",
    shaves=6,
    version="2024.0",
    optimizer_params=[
        "--input_shape=[1,28,28,1]",
        "--mean_values=[0]",
        "--scale_values=[255]"
    ]
)

print(f"Success! Blob saved at: {blob_path}")