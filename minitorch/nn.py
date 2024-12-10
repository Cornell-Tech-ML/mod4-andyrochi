from typing import Tuple

from .tensor import Tensor
from .tensor_functions import Function, rand
from .autodiff import Context

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    input = input.permute(0, 1, 2, 4, 3, 5)
    input = input.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    return input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to an image tensor

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # TODO: Implement for Task 4.3.
    input, new_height, new_width = tile(input, kernel)
    return (
        input.mean(dim=4)
        .contiguous()
        .view(input.shape[0], input.shape[1], new_height, new_width)
    )


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Max function"""
        ctx.save_for_backward(t1, dim)
        return t1.f.max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Max function"""
        t1, dim = ctx.saved_values
        return grad_output * argmax(t1, dim), 0.0


def max(t1: Tensor, dim: int) -> Tensor:
    """Max function"""
    return Max.apply(t1, t1._ensure_tensor(dim))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: Tensor
        dim: Dimension to compute argmax

    Returns:
    -------
        Tensor of indices

    """
    max_val = max(input, dim)
    return input == max_val


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: Tensor
        dim: Dimension to compute softmax along

    Returns:
    -------
        Tensor of softmax values

    """
    exp = input.exp()
    return exp / exp.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: Tensor
        dim: Dimension to compute logsoftmax along

    Returns:
    -------
        Tensor of log softmax values

    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to an image tensor

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    input, new_height, new_width = tile(input, kernel)
    return (
        max(input, dim=4)
        .contiguous()
        .view(input.shape[0], input.shape[1], new_height, new_width)
    )


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: Tensor to apply dropout to
        rate: Probability of dropping a position (0.0 to 1.0)
        ignore: If True, disable dropout and return input unchanged

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore:
        return input

    # Use tensor backend's random function instead of direct random method
    rand_val = rand(input.shape, backend=input.backend)
    mask = rand_val > rate

    scale = 1.0 / (1.0 - rate) if rate != 1.0 else 1.0
    return mask * input * scale
