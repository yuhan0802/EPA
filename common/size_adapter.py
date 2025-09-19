import math

from torch import nn


def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)


class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]


class BatchSizeAdapter(object):
    """Converts size of input to standard size for batched data.
    This class allows padding and unpadding of input batches
    where each key in the batch dictionary contains a tensor
    with different channels but same spatial dimensions.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        """Finds the closest multiple of the minimum size greater than or equal to the given size."""
        return ((size + self._minimum_size - 1) // self._minimum_size) * self._minimum_size

    def pad(self, batch):
        """Pads all tensors in the batch to the closest multiple of the minimum size."""
        sample_tensor = next(iter(batch.values()))
        height, width = sample_tensor.size()[-2:]  # 获取 H, W

        self._pixels_pad_to_height = self._closest_larger_multiple_of_minimum_size(height) - height
        self._pixels_pad_to_width = self._closest_larger_multiple_of_minimum_size(width) - width

        padding_layer = nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))
        padded_batch = {key: padding_layer(tensor) for key, tensor in batch.items()}

        return padded_batch

    def batch_unpad(self, batch):
        return {
            key: tensor[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]
            for key, tensor in batch.items()
        }

    def unpad(self, network_output):
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]


class ImgAndEventSizeAdapter(object):

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, imgs, voxels, mask):
        height, width = imgs.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        imgs = nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(imgs)
        voxels = nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(voxels)
        mask = nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(mask)
        return imgs, voxels, mask

    def unpad(self, pred):
        return pred[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]
