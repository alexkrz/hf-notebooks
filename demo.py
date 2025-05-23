import torch


def main():
    # Assign device where code is executed
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Neural Engine (MPS)
    else:
        device = torch.device("cpu")  # Default to CPU

    print("device:", device)


if __name__ == "__main__":
    main()
