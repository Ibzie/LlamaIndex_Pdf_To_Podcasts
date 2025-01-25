import torch

def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    test_cuda()