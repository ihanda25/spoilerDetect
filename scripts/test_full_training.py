"""Check coverage and review-level pooling before the long training run."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from full_training import Reviews, collate, review_logits


class WindowTests(unittest.TestCase):
    def dataset(self, length):
        dataset = Reviews.__new__(Reviews)
        dataset.tokens = np.arange(length, dtype=np.int32) + 200
        dataset.offsets = np.array([0, length])
        return dataset

    def test_all_tokens_and_boundaries(self):
        for length in [1, 509, 510, 511, 956, 957, 3317]:
            dataset = self.dataset(length)
            chunks = dataset.chunks(0)
            recovered = chunks[0][1:-1]
            for chunk in chunks[1:]:
                recovered += chunk[65:-1]
            self.assertEqual(recovered, dataset.tokens.tolist())
            self.assertTrue(all(len(c) <= 512 and c[0] == 101 and c[-1] == 102 for c in chunks))
            inputs, sizes = collate(dataset, [0])
            self.assertEqual(sizes, [len(chunks)])
            self.assertEqual(inputs['attention_mask'].sum().item(), sum(map(len, chunks)))

    def test_pooling_uses_one_score_per_review(self):
        margins = torch.tensor([.1, 2., -.5], requires_grad=True)
        class FakeModel:
            def __call__(self, **inputs):
                return SimpleNamespace(logits=torch.stack([margins*0, margins], dim=1))
        inputs = {'input_ids': torch.zeros((3,2), dtype=torch.long)}
        pooled = review_logits(FakeModel(), inputs, [2,1], torch.device('cpu'))
        self.assertEqual(pooled.tolist(), [2., -.5])
        pooled.sum().backward()
        self.assertEqual(margins.grad.tolist(), [0., 1., 1.])


if __name__ == '__main__':
    unittest.main()
