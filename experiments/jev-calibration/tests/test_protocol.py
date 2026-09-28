import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from protocol import extract_input_ids  # noqa: E402


class TensorLike:
    def tolist(self):
        return [7, 8]


class ProtocolTest(unittest.TestCase):
    def test_batch_encoding_shape(self):
        self.assertEqual(extract_input_ids({"input_ids": [1, 2, 3]}), [1, 2, 3])

    def test_tensor_like_shape(self):
        self.assertEqual(extract_input_ids(TensorLike()), [7, 8])

    def test_rejects_rendered_chat_text(self):
        with self.assertRaises(TypeError):
            extract_input_ids("<|im_start|>user")


if __name__ == "__main__":
    unittest.main()
