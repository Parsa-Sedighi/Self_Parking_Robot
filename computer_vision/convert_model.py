import blobconverter
import certifi
import os

# This forces the script to use the fresh certificates we just installed
os.environ['SSL_CERT_FILE'] = certifi.where()

model_path = "computer_vision/model/frozen_graph.pb"

try:
    blob_path = blobconverter.from_tf(
        frozen_pb=model_path,
        data_type="FP16",
        shaves=6,
        optimizer_params=["--mean_values=[0]", "--scale_values=[255]", "--input_shape=[1,28,28,1]"],
        compile_params=["-ip U8"],
        output_dir="computer_vision/model/"
    )
    print(f"✅ SUCCESS! Blob saved at: {blob_path}")
except Exception as e:
    print(f"❌ Still failing: {e}")