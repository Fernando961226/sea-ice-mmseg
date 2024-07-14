'''
No@
July 4th, 2024
'''
from typing import Iterator, Sized
import torch
from mmengine.registry import DATA_SAMPLERS
from mmengine.dataset.dataset_wrapper import ConcatDataset
from mmengine.dataset.sampler import InfiniteSampler


@DATA_SAMPLERS.register_module()
class WeightedInfiniteSampler(InfiniteSampler):
    """
    Samples elements from a set of concatenated dataset``[dataset0, dataset1,...]`` 
    with given probabilities (weights).
    The weight for a particular dataset is inverselly proportional to the number of samples.

    Args:
        dataset (Sized): The dataset.
        use_weights = If True use weights to generate samples. If False, use InfiniteSampler
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """
    def __init__(self,
                 dataset: Sized,
                 use_weights: bool = True, 
                 replacement: bool = True, 
                 **kwargs) -> None:
        super().__init__(dataset, **kwargs)

        if use_weights:
            assert isinstance(dataset, ConcatDataset), \
                'The dataset must be ConcatDataset type to use this sampler'

            self.num_samples = dataset.cumulative_sizes[-1]
            self.replacement = replacement

            sizes = torch.as_tensor(dataset.cumulative_sizes, dtype=torch.double)
            sizes = torch.cat((sizes[:1], sizes[1:] - sizes[:-1]))
            # sizes[1:] -= sizes[:-1]
            weights = []
            for sz in sizes:
                weights.extend([1/sz]*int(sz))
            self.weights = torch.as_tensor(weights, dtype=torch.double)

            self.indices = self._indices_of_rank()

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            rand_tensor = torch.multinomial(self.weights, self.num_samples, 
                                            self.replacement, generator=g)
            yield from iter(rand_tensor.tolist())

