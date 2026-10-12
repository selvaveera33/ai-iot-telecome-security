import unittest
from data_generator import FEATURES, generate_dataset


class DatasetGeneratorTests(unittest.TestCase):
    def test_reproducible_and_columns(self):
        a, b = generate_dataset(2000), generate_dataset(2000)
        self.assertTrue(a.equals(b))
        self.assertEqual(list(a.columns), FEATURES + ["label"])
        self.assertEqual(len(a), 2000)
        self.assertEqual(set(a.label.unique()), {0, 1})

    def test_small_size_rejected(self):
        with self.assertRaises(ValueError):
            generate_dataset(1)
