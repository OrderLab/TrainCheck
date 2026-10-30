"""Backend routing and numeric parity for the existing tensor-hash algorithm."""

import unittest
from collections.abc import Sequence
from unittest.mock import patch

import torch

from traincheck.instrumentor import dumper
from traincheck.instrumentor.proxy_wrapper import hash as tensor_hashing


def reference_hash(values: Sequence[int]) -> int:
    """Independent Python-integer recurrence with explicit signed-64 overflow."""
    accumulator = 0
    for value in values:
        accumulator = (accumulator * 6364136223846793005 + value + 1) % (2**64)
    return accumulator - 2**64 if accumulator >= 2**63 else accumulator


def first_quantized_row(tensor: torch.Tensor) -> list[int]:
    """Preserve the historical quantization and first-row selection contract."""
    cpu = tensor.detach().cpu()
    if cpu.is_floating_point():
        cpu = (cpu * 1e8).to(torch.int64)
    else:
        cpu = cpu.to(torch.int64)
    if cpu.ndim == 0:
        return [int(cpu.item())]
    if cpu.ndim == 1:
        return [int(value) for value in cpu.tolist()]
    return [int(value) for value in cpu.reshape(-1, cpu.shape[-1])[0].tolist()]


class CudaTaggedTensor(torch.Tensor):
    """Exercise routing without claiming that this CPU-backed fixture is a GPU."""

    is_cuda: bool = True


class TensorDumpRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        selected = patch.dict(
            dumper.tensor_dump_format,
            dump_tensor_hash=True,
            dump_tensor_stats=False,
            dump_tensor_full=False,
        )
        selected.start()
        self.addCleanup(selected.stop)

    def test_cuda_tag_routes_to_cuda_without_cpu_fallback(self) -> None:
        value = torch.zeros(1).as_subclass(CudaTaggedTensor)
        with patch.object(dumper, "tensor_hash", return_value=37) as calculate:
            self.assertEqual(dumper.dump_tensor(value), 37)
        calculate.assert_called_once_with(value, with_parallel=True, with_cuda=True)

    def test_non_cuda_tensor_keeps_cpu_hash_route(self) -> None:
        value = torch.zeros(1)
        with patch.object(dumper, "tensor_hash", return_value=41) as calculate:
            self.assertEqual(dumper.dump_tensor(value), 41)
        calculate.assert_called_once_with(value, with_parallel=True, with_cuda=False)

    def test_cuda_hash_failure_propagates_without_silent_fallback(self) -> None:
        value = torch.zeros(1).as_subclass(CudaTaggedTensor)
        error = RuntimeError("CUDA hash failed")
        with patch.object(dumper, "tensor_hash", side_effect=error) as calculate:
            with self.assertRaises(RuntimeError) as caught:
                dumper.dump_tensor(value)
        self.assertIs(caught.exception, error)
        calculate.assert_called_once_with(value, with_parallel=True, with_cuda=True)

    def test_other_dump_modes_and_non_tensors_keep_their_behavior(self) -> None:
        value = torch.tensor([1.0, 2.0])
        with (
            patch.dict(
                dumper.tensor_dump_format, dump_tensor_hash=False, dump_tensor_full=True
            ),
            patch.object(dumper, "tensor_hash") as calculate,
        ):
            self.assertEqual(dumper.dump_tensor(value), [1.0, 2.0])
            self.assertIsNone(dumper.dump_tensor(object()))
            calculate.assert_not_called()
        cached = object()
        with (
            patch.dict(dumper.tensor_dump_format, dump_tensor_stats=True),
            patch.object(dumper, "tensor_stats", return_value=cached),
            patch.object(dumper, "tensor_hash") as calculate,
        ):
            self.assertIs(dumper.dump_tensor(value), cached)
            calculate.assert_not_called()

    def test_cpu_hash_matches_independent_recurrence_without_mutation(self) -> None:
        for shape in ((), (4,), (3, 4), (2, 3, 4)):
            elements = 1
            for dimension in shape:
                elements *= dimension
            value = (torch.arange(elements, dtype=torch.int64) - 3).reshape(shape)
            original = value.clone()
            rng = torch.get_rng_state().clone()
            with self.subTest(shape=shape):
                self.assertEqual(
                    tensor_hashing.tensor_hash(value, with_cuda=False),
                    reference_hash(first_quantized_row(value)),
                )
                self.assertTrue(torch.equal(value, original))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))


@unittest.skipUnless(torch.cuda.is_available(), "actual CUDA device required")
class CudaHashValueTests(unittest.TestCase):
    def check_value(self, value: torch.Tensor) -> None:
        self.assertTrue(value.is_cuda)
        original = value.clone()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(value.device).clone()
        expected = reference_hash(first_quantized_row(value))
        cpu_hash = tensor_hashing.tensor_hash(value, with_cuda=False)
        cuda_hash = tensor_hashing.tensor_hash(value, with_cuda=True)
        self.assertEqual(cuda_hash, expected)
        self.assertEqual(cuda_hash, cpu_hash)
        self.assertTrue(torch.equal(value, original))
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(value.device), cuda_rng))

    def test_cuda_scalar_hashes_match_cpu_for_supported_dtypes(self) -> None:
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            for scalar in (0, 1, -1) if dtype != torch.uint8 else (0, 1, 255):
                # Small representable half values avoid quantization to infinity.
                value = scalar / 16384 if dtype == torch.float16 else scalar
                with self.subTest(dtype=dtype, scalar=value):
                    self.check_value(torch.tensor(value, dtype=dtype, device="cuda"))

    def test_cuda_vector_matrix_and_3d_match_the_same_recurrence(self) -> None:
        for shape in ((4,), (3, 4), (2, 3, 4)):
            elements = 1
            for dimension in shape:
                elements *= dimension
            for dtype in (torch.int64, torch.float32, torch.float64, torch.bfloat16):
                with self.subTest(shape=shape, dtype=dtype):
                    value = (torch.arange(elements, device="cuda") - 3).to(dtype)
                    self.check_value(value.reshape(shape))

    def test_cuda_noncontiguous_tensor_and_empty_vector_keep_existing_results(
        self,
    ) -> None:
        value = torch.arange(24, dtype=torch.int64, device="cuda").reshape(4, 6)
        self.check_value(value[:, ::2])
        self.check_value(value.t())
        self.check_value(torch.empty(0, dtype=torch.float32, device="cuda"))

    def test_real_cuda_dumper_uses_cuda_hash_and_preserves_gradients(self) -> None:
        value = torch.tensor(-0.5, device="cuda", requires_grad=True)
        value.grad = torch.tensor(2.0, device="cuda")
        original_gradient = value.grad.clone()
        with (
            patch.dict(
                dumper.tensor_dump_format,
                dump_tensor_hash=True,
                dump_tensor_stats=False,
                dump_tensor_full=False,
            ),
            patch.object(
                tensor_hashing,
                "hash_tensor_cuda",
                wraps=tensor_hashing.hash_tensor_cuda,
            ) as cuda_backend,
            patch.object(
                tensor_hashing,
                "hash_tensor_cpu",
                side_effect=AssertionError("CPU fallback"),
            ),
        ):
            self.assertEqual(
                dumper.dump_tensor(value), reference_hash(first_quantized_row(value))
            )
            cuda_backend.assert_called_once()
        self.assertTrue(torch.equal(value.grad, original_gradient))


if __name__ == "__main__":
    unittest.main()
