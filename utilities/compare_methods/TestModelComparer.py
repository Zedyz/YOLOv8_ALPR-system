import unittest
from ModelComparer import ModelComparer


class TestModalComparer(unittest.TestCase):
    def setUp(self):
        self.model_comparer = ModelComparer("ATU9776")

    def test_total_equality(self):
        test_cases = [
            ('ATU9776', True),
            ('ABAATU9776', True),
            ('ATU9776ATU', True),
            ('ATUATU9776ATU', True),
            ('BCD6543', False),
            ('ATU6734', False),
            ('AT_9776', False)
        ]

        for to_compare, expected in test_cases:
            with self.subTest(to_compare=to_compare):
                is_equal, _ = self.model_comparer.total_equality(to_compare)
                self.assertEqual(is_equal, expected)

    def test_compare(self):
        test_cases = [
            ('ATU9776', True)
        ]


if __name__ == '__main__':
    unittest.main()
