import numpy as np

from torch.utils.data import BatchSampler

class UniformYawBatchSampler(BatchSampler):

    def __init__(self, samples_dict, batch_size=4, drop_last=False):
        self.samples_dict = {k:v.copy() for k, v in samples_dict.copy().items() if len(v) != 0}
        self.samples_keys = list(self.samples_dict.keys())
        self.samples = []
        for key in self.samples_dict.keys():
            self.samples += self.samples_dict[key]
            np.random.shuffle(self.samples_dict[key])

        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        while True:
            if len(self.samples_keys) > 0:
                idx_1D = np.random.choice(self.samples_keys, 1)[0]
                if len(self.samples_dict[idx_1D]) > 0:
                    np.random.shuffle(self.samples_dict[idx_1D])
                    element = self.samples_dict[idx_1D].pop()
                    batch.append(element)
                else:
                    self.samples_keys.remove(idx_1D)
            else:
                break

            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.samples) // self.batch_size
        else:
            return (len(self.samples) + self.batch_size - 1) // self.batch_size