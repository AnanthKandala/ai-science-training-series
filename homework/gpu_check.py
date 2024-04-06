import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available CUDA devices
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {device_count}")

    # List details of each CUDA device
    for i in range(device_count):
        device = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {device.name}")
else:
    print("CUDA is not available. Running on CPU.")
