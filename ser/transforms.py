from torchvision import transforms


# torch transforms

def transforms1():
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return ts
