"""TurboQuant vLLM integration: quantization config + TQ3 checkpoint loader.

Three roles:
1. Register ``TurboQuantConfig`` with ``--quantization turboquant`` so
   vLLM allocates model weights on meta device (zero GPU at init).
2. Online quant methods (``TurboQuantOnlineLinearMethod``,
   ``TurboQuantOnlineMoEMethod``) compress bf16 → TQ3 per-layer after
   weight loading, keeping peak GPU memory at ~1 layer bf16.
3. Patch ``DefaultModelLoader.get_all_weights`` to pass through native
   TQ3 checkpoints (``.tq_packed`` / ``.tq_norms``) and bind packed
   buffers directly, skipping bf16 decompression entirely.

``TurboQuantConfig`` MUST live at module top level. cloudpickle
serializes closure-defined classes by value, transitively pulling in
``torch.ops.turboquant.*`` and crashing vLLM worker startup with
``cannot pickle '_OpNamespace'`` (issue #39).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_TQ_PARAM_ATTRS = ("output_dim", "input_dim", "packed_dim", "packed_factor", "is_metadata")

_PACKED_SHARD_ORDER = {
    "q": 0,
    "k": 1,
    "v": 2,
    "gate": 0,
    "up": 1,
    "w1": 0,
    "w2": 1,
    "w3": 2,
}


def _extract_weight_name(args: tuple, kwargs: dict) -> str | None:
    for key in ("weight_name", "name", "param_name"):
        val = kwargs.get(key)
        if isinstance(val, str):
            return val
    for arg in args[2:]:
        if isinstance(arg, str):
            return arg
    return None


def _extract_shard_id(args: tuple, kwargs: dict) -> object | None:
    if "shard_id" in kwargs:
        return kwargs["shard_id"]
    if "expert_id" in kwargs:
        return None
    if len(args) >= 3:
        if isinstance(args[2], str) and ".tq_" in args[2]:
            return None
        return args[2]
    return None


def _ordered_shard_keys(keys: list[object]) -> list[object]:
    if not keys:
        return []
    if all(isinstance(k, str) and k in _PACKED_SHARD_ORDER for k in keys):
        return sorted(keys, key=lambda k: _PACKED_SHARD_ORDER[str(k)])
    if all(isinstance(k, int) for k in keys):
        return sorted(keys)
    return list(keys)


def _record_linear_packed(layer: nn.Module, shard_id: object | None, tensor: torch.Tensor, is_norms: bool) -> None:
    if not hasattr(layer, "_tq_packed_shards"):
        layer._tq_packed_shards = {}
        layer._tq_norms_shards = {}
        layer._tq_shard_order = []
    key = shard_id if shard_id is not None else len(layer._tq_shard_order)
    if key not in layer._tq_shard_order:
        layer._tq_shard_order.append(key)
    if is_norms:
        layer._tq_norms_shards[key] = tensor
    else:
        layer._tq_packed_shards[key] = tensor
    layer._tq_has_packed = True


def _consume_linear_packed(layer: nn.Module, n_groups: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not getattr(layer, "_tq_has_packed", False):
        return None
    packed_shards = getattr(layer, "_tq_packed_shards", {})
    norms_shards = getattr(layer, "_tq_norms_shards", {})
    shard_order = _ordered_shard_keys(list(getattr(layer, "_tq_shard_order", [])))
    if not packed_shards or not norms_shards:
        missing_kinds = []
        if not packed_shards:
            missing_kinds.append(".tq_packed")
        if not norms_shards:
            missing_kinds.append(".tq_norms")
        raise RuntimeError(
            f"TQ3 packed load expected shards for {', '.join(missing_kinds)}, but none were loaded."
        )
    missing = [key for key in shard_order if key not in packed_shards or key not in norms_shards]
    if missing:
        raise RuntimeError(f"TQ3 packed load missing shards: {missing}")
    packed = torch.cat([packed_shards[key] for key in shard_order], dim=0)
    norms = torch.cat([norms_shards[key] for key in shard_order], dim=0)
    if norms.shape[1] != n_groups:
        raise RuntimeError(
            f"TQ3 packed load: norms shape {tuple(norms.shape)} does not match n_groups={n_groups}"
        )
    layer._tq_packed_shards.clear()
    layer._tq_norms_shards.clear()
    layer._tq_shard_order.clear()
    return packed, norms

# vLLM is an optional dependency — the package imports cleanly without
# it (Mac/MLX-only paths). Class definitions below are guarded on the
# imported symbols being non-None.
try:
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
except ImportError:
    LinearBase = None  # type: ignore[assignment,misc]
    QuantizationConfig = object  # type: ignore[assignment,misc]
    QuantizeMethodBase = object  # type: ignore[assignment,misc]

try:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
except ImportError:
    FusedMoEMethodBase = object  # type: ignore[assignment,misc]
    UnquantizedFusedMoEMethod = None  # type: ignore[assignment,misc]


# Shared scratch pool across all FusedMoE layers — only one MoE layer
# runs at a time during forward, so one set of bf16 decompression
# buffers is enough. Per-layer pools would consume 78 × ~5 GB = 390 GB
# and defeat compression entirely.
_shared_moe_scratch_pool = None


# ── TurboQuantConfig: registered as `--quantization turboquant` ──

if LinearBase is not None:

    class TurboQuantConfig(QuantizationConfig):
        """Config for TurboQuant weight quantization (TQ3/TQ4)."""

        def __init__(self, bits: int = 3, group_size: int = 128, sensitive_bits: int | None = None):
            super().__init__()
            if bits not in (2, 3, 4):
                raise ValueError(f"turboquant bits must be 2, 3, or 4; got {bits}")
            if group_size <= 0 or group_size % 8 != 0:
                raise ValueError(f"turboquant group_size must be a positive multiple of 8; got {group_size}")
            if sensitive_bits is not None and sensitive_bits not in (2, 3, 4):
                raise ValueError(f"turboquant sensitive_bits must be 2, 3, or 4 or None; got {sensitive_bits}")
            self.bits = bits
            self.group_size = group_size
            self.sensitive_bits = sensitive_bits

        def __repr__(self) -> str:
            return (
                f"TurboQuantConfig(bits={self.bits}, group_size={self.group_size}, "
                f"sensitive_bits={self.sensitive_bits})"
            )

        def get_name(self) -> str:
            return "turboquant"

        def get_supported_act_dtypes(self) -> list[torch.dtype]:
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 70  # Volta and newer

        @staticmethod
        def get_config_filenames() -> list[str]:
            return ["tq_config.json", "quantize_config.json"]

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "TurboQuantConfig":
            bits = cls.get_from_keys_or(config, ["bits"], 3)
            group_size = cls.get_from_keys_or(config, ["group_size"], 128)
            sensitive_bits = cls.get_from_keys_or(config, ["sensitive_bits"], None)
            return cls(bits=bits, group_size=group_size, sensitive_bits=sensitive_bits)

        def get_quant_method(self, layer: nn.Module, prefix: str) -> "QuantizeMethodBase | None":
            if isinstance(layer, LinearBase):
                return TurboQuantOnlineLinearMethod(self.bits, self.group_size)
            try:
                from vllm.model_executor.layers.fused_moe import FusedMoE

                if isinstance(layer, FusedMoE) and TurboQuantOnlineMoEMethod is not None:
                    return TurboQuantOnlineMoEMethod(
                        self.bits,
                        self.group_size,
                        layer.moe_config,
                    )
            except ImportError:
                pass
            return None

else:
    TurboQuantConfig = None  # type: ignore[assignment,misc]


# ── Online Linear quant method (meta-device init, per-layer compression) ──

if LinearBase is not None:

    class TurboQuantOnlineLinearMethod(QuantizeMethodBase):
        """Meta-device init + per-layer TQ3 compression for Linear layers.

        Allocates bf16 weight on meta device (zero GPU at init). After
        weight loading materializes the bf16 on GPU, compress to TQ3
        packed format. For native TQ3 checkpoints, packed buffers are
        bound directly without bf16 decompression.
        """

        uses_meta_device: bool = True

        def __init__(self, bits: int, group_size: int):
            self.bits = bits
            self.group_size = group_size

        def create_weights(
            self,
            layer: nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ):
            from vllm.model_executor.model_loader.reload.layerwise import (
                initialize_online_processing,
            )
            from vllm.model_executor.parameter import ModelWeightParameter

            output_size_per_partition = sum(output_partition_sizes)
            weight_loader = extra_weight_attrs.get("weight_loader")

            if weight_loader is not None:
                from turboquant_vllm.weight_quant import packed_group_bytes, padded_size

                padded_in, n_groups = padded_size(input_size_per_partition, self.group_size)
                bytes_per_group = packed_group_bytes(self.bits, self.group_size)

                def _tq_weight_loader(param, loaded_weight, *args, **kwargs):
                    if isinstance(loaded_weight, torch.Tensor):
                        weight_name = _extract_weight_name(args, kwargs)
                        shard_id = _extract_shard_id(args, kwargs)
                        if weight_name is not None:
                            if weight_name.endswith(".tq_packed"):
                                _record_linear_packed(layer, shard_id, loaded_weight, is_norms=False)
                                return True
                            if weight_name.endswith(".tq_norms"):
                                _record_linear_packed(layer, shard_id, loaded_weight, is_norms=True)
                                return True
                        if (
                            loaded_weight.dtype == torch.uint8
                            and loaded_weight.ndim == 2
                            and loaded_weight.shape[1] == bytes_per_group
                        ):
                            _record_linear_packed(layer, shard_id, loaded_weight, is_norms=False)
                            return True
                        if (
                            loaded_weight.is_floating_point()
                            and loaded_weight.ndim == 2
                            and loaded_weight.shape[1] == n_groups
                        ):
                            _record_linear_packed(layer, shard_id, loaded_weight, is_norms=True)
                            return True
                    return weight_loader(param, loaded_weight, *args, **kwargs)

                weight_loader = _tq_weight_loader

            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    device="meta",
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            initialize_online_processing(layer)

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return

            from turboquant_vllm.weight_quant import (
                _ensure_triton_backends,
                _get_cuda_module,
                _get_quantizer,
                _tq_fused_gemm_fn,
                _tq_fwht_input_fn,
                _triton_available,
                pack_indices,
                packed_group_bytes,
                padded_size,
            )

            weight = layer.weight.data
            bits = self.bits
            group_size = self.group_size

            out_dim, in_dim = weight.shape
            padded_in, n_groups = padded_size(in_dim, group_size)

            packed_pair = _consume_linear_packed(layer, n_groups)
            if packed_pair is not None:
                packed, norms = packed_pair
                if norms.shape[0] != out_dim:
                    out_dim = norms.shape[0]

                layer.weight.data = torch.empty(0, device=packed.device, dtype=weight.dtype)
                layer.register_buffer("tq_packed_weight", packed)
                layer.register_buffer("tq_norms", norms)
                quantizer = _get_quantizer(group_size, bits, str(packed.device))
                layer.register_buffer("tq_signs1", quantizer.signs1)
                layer.register_buffer("tq_signs2", quantizer.signs2)
                layer.register_buffer("tq_centroids", quantizer.centroids)
                arch_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(packed.device)[0] >= 8
                if bits == 3 and group_size == 128 and arch_ok:
                    bytes_per_group = packed_group_bytes(bits, group_size)
                    layer.register_buffer(
                        "tq_packed_bs1",
                        packed.view(out_dim * n_groups, bytes_per_group),
                    )
                    layer.register_buffer("tq_norms_bf16", norms.to(torch.bfloat16))
                    layer.register_buffer(
                        "tq_centroids_bf16",
                        quantizer.centroids.to(torch.bfloat16),
                    )
                layer.tq_in_features = in_dim
                layer.tq_out_features = out_dim
                layer.tq_padded_in = padded_in

                _ensure_triton_backends()
                _get_cuda_module()
                if _triton_available:
                    layer._tq_primary_fn = _tq_fwht_input_fn if out_dim >= 4096 else _tq_fused_gemm_fn
                    layer._tq_fallback_fn = _tq_fused_gemm_fn if out_dim >= 4096 else _tq_fwht_input_fn
                else:
                    layer._tq_primary_fn = None

                layer._already_called_process_weights_after_loading = True
                return

            if padded_in > in_dim:
                padded = torch.zeros(
                    out_dim,
                    padded_in,
                    dtype=weight.dtype,
                    device=weight.device,
                )
                padded[:, :in_dim] = weight
            else:
                padded = weight

            grouped = padded.reshape(-1, group_size)
            quantizer = _get_quantizer(group_size, bits, str(weight.device))
            indices, norms_raw = quantizer.quantize(grouped, norm_correction=True)
            packed = pack_indices(indices, bits)
            norms = norms_raw.reshape(out_dim, n_groups)

            # Keep weight for vLLM's MLA/attention post-processing,
            # but zero it to free most GPU memory. Full deletion breaks
            # MLAAttention.process_weights_after_loading which accesses
            # sub-layer weights after our quant method runs.
            layer.weight.data = torch.empty(0, device=weight.device, dtype=weight.dtype)
            layer.register_buffer("tq_packed_weight", packed)
            layer.register_buffer("tq_norms", norms)
            layer.register_buffer("tq_signs1", quantizer.signs1)
            layer.register_buffer("tq_signs2", quantizer.signs2)
            layer.register_buffer("tq_centroids", quantizer.centroids)
            # Pre-cast bf16 companions consumed by the bs=1 CUDA GEMV fast path.
            # Casting once at load time avoids per-decode-step HBM traffic.
            # Gate registration on the arch requirement so apply()'s fast-path
            # check collapses to a single hasattr() rather than a per-call
            # cudaGetDeviceProperties query.
            arch_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(weight.device)[0] >= 8
            if bits == 3 and group_size == 128 and arch_ok:
                bytes_per_group = group_size * bits // 8
                layer.register_buffer(
                    "tq_packed_bs1",
                    packed.view(out_dim * n_groups, bytes_per_group),
                )
                layer.register_buffer("tq_norms_bf16", norms.to(torch.bfloat16))
                layer.register_buffer(
                    "tq_centroids_bf16",
                    quantizer.centroids.to(torch.bfloat16),
                )
            layer.tq_in_features = in_dim
            layer.tq_out_features = out_dim
            layer.tq_padded_in = padded_in

            # Cache dispatch — must run before CUDA graph capture
            _ensure_triton_backends()
            _get_cuda_module()
            if _triton_available:
                layer._tq_primary_fn = _tq_fwht_input_fn if out_dim >= 4096 else _tq_fused_gemm_fn
                layer._tq_fallback_fn = _tq_fused_gemm_fn if out_dim >= 4096 else _tq_fwht_input_fn
            else:
                layer._tq_primary_fn = None

            layer._already_called_process_weights_after_loading = True
            del weight, padded, grouped, indices, norms_raw

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # Pad input if in_dim was not a multiple of group_size
            if x.shape[-1] != layer.tq_padded_in:
                x = torch.nn.functional.pad(x, (0, layer.tq_padded_in - x.shape[-1]))

            # Route TQ3 bf16 through a runtime-dispatching custom op so the
            # bs=1 CUDA GEMV gets captured inside each size-specific CUDA
            # graph. Dynamo traces the model once (batch >> 1 on
            # profile_run) and would specialize a Python-level M==1 branch
            # against that shape, so the branch must live inside the op.
            if bias is None and self.bits == 3 and x.dtype == torch.bfloat16 and hasattr(layer, "tq_packed_bs1"):
                return torch.ops.turboquant.tq3_apply(
                    x,
                    layer.tq_packed_weight,
                    layer.tq_norms,
                    layer.tq_signs1,
                    layer.tq_signs2,
                    layer.tq_centroids,
                    layer.tq_packed_bs1,
                    layer.tq_norms_bf16,
                    layer.tq_centroids_bf16,
                    self.group_size,
                    self.bits,
                )

            if layer._tq_primary_fn is not None:
                args = (
                    x,
                    layer.tq_packed_weight,
                    layer.tq_norms,
                    layer.tq_signs1,
                    layer.tq_signs2,
                    layer.tq_centroids,
                )
                try:
                    return layer._tq_primary_fn(
                        *args,
                        group_size=self.group_size,
                        bits=self.bits,
                        bias=bias,
                    )
                except (ValueError, RuntimeError) as e:
                    logger.warning("TurboQuant primary kernel failed, using fallback: %s", e)
                    return layer._tq_fallback_fn(
                        *args,
                        group_size=self.group_size,
                        bits=self.bits,
                        bias=bias,
                    )

            # CPU/CUDA fallback
            from turboquant_vllm.weight_quant import _get_quantizer, unpack_indices

            indices = unpack_indices(
                layer.tq_packed_weight,
                self.bits,
                self.group_size,
            )
            norms_flat = layer.tq_norms.reshape(-1)
            quantizer = _get_quantizer(
                self.group_size,
                self.bits,
                str(x.device),
            )
            w_groups = quantizer.dequantize(indices, norms_flat)
            w_deq = w_groups.reshape(
                layer.tq_out_features,
                layer.tq_padded_in,
            ).to(x.dtype)
            output = torch.matmul(x, w_deq.t())
            if bias is not None:
                output = output + bias
            return output

else:
    TurboQuantOnlineLinearMethod = None  # type: ignore[assignment,misc]


# ── MoE online method ──


def _materialize_and_process(
    layer,
    buffer,
    orig_loaders,
    param_shapes,
    param_dtypes,
    method,
):
    """Materialize meta params on GPU, replay buffered loads, compress."""
    # 1. Materialize meta → real tensors on GPU
    for name, param in list(layer.named_parameters(recurse=False)):
        if param.device == torch.device("meta") and name in param_shapes:
            real = torch.empty(
                param_shapes[name],
                dtype=param_dtypes[name],
                device="cuda",
            )
            real_param = torch.nn.Parameter(real, requires_grad=False)
            if name in orig_loaders:
                real_param.weight_loader = orig_loaders[name]
            for attr in _TQ_PARAM_ATTRS:
                if hasattr(param, attr):
                    setattr(real_param, attr, getattr(param, attr))
            delattr(layer, name)
            layer.register_parameter(name, real_param)

    # 2. Replay all buffered weight_loader calls
    for pname, args, kwargs in buffer:
        loader = orig_loaders.get(pname)
        if loader is not None:
            param = getattr(layer, pname)
            new_args = (param,) + args[1:]
            loader(*new_args, **kwargs)
    buffer.clear()

    # 3. Kernel setup + compress
    method._do_compress(layer)


def _materialize_packed_moe(
    layer: nn.Module,
    orig_loaders: dict[str, Any],
    param_shapes: dict[str, tuple],
    param_dtypes: dict[str, torch.dtype],
    device: torch.device,
    packed_param_names: set[str],
) -> None:
    for name, param in list(layer.named_parameters(recurse=False)):
        if param.device == torch.device("meta") and name in param_shapes:
            if name in packed_param_names:
                real = torch.empty(0, dtype=param_dtypes[name], device=device)
            else:
                real = torch.empty(param_shapes[name], dtype=param_dtypes[name], device=device)
            real_param = torch.nn.Parameter(real, requires_grad=False)
            if name in orig_loaders:
                real_param.weight_loader = orig_loaders[name]
            for attr in _TQ_PARAM_ATTRS:
                if hasattr(param, attr):
                    setattr(real_param, attr, getattr(param, attr))
            delattr(layer, name)
            layer.register_parameter(name, real_param)


if UnquantizedFusedMoEMethod is not None and LinearBase is not None:

    class TurboQuantOnlineMoEMethod(FusedMoEMethodBase):
        """Meta-device MoE: compress after loading, decompress per forward.

        The MoE kernel is initialized by the underlying unquantized
        method's ``process_weights_after_loading``. After compression,
        ``apply()`` decompresses into a shared scratch pool and
        delegates to the unquantized method (which has the kernel).
        """

        uses_meta_device: bool = True

        def __init__(self, bits: int, group_size: int, moe_config: Any):
            super().__init__(moe_config)
            self.bits = bits
            self.group_size = group_size
            self._unquant = UnquantizedFusedMoEMethod(moe_config)
            self._pool = None
            self._w13_c = None
            self._w2_c = None

        def create_weights(self, layer: nn.Module, **kwargs):
            self._unquant.create_weights(layer, **kwargs)

            # Compute expected total numel for completion tracking
            total_numel = sum(p.numel() for p in layer.parameters(recurse=False))

            # Save original weight_loaders + shapes BEFORE meta move
            orig_loaders: dict[str, Any] = {}
            param_shapes: dict[str, tuple] = {}
            param_dtypes: dict[str, torch.dtype] = {}
            for name, param in list(layer.named_parameters(recurse=False)):
                if hasattr(param, "weight_loader"):
                    orig_loaders[name] = param.weight_loader
                param_shapes[name] = tuple(param.shape)
                param_dtypes[name] = param.dtype

            # Move parameters to meta device (zero GPU at init)
            for name, param in list(layer.named_parameters(recurse=False)):
                if param.device != torch.device("meta"):
                    meta_param = torch.nn.Parameter(
                        torch.empty_like(param, device="meta"),
                        requires_grad=False,
                    )
                    if hasattr(param, "weight_loader"):
                        meta_param.weight_loader = param.weight_loader
                    for attr in _TQ_PARAM_ATTRS:
                        if hasattr(param, attr):
                            setattr(meta_param, attr, getattr(param, attr))
                    delattr(layer, name)
                    layer.register_parameter(name, meta_param)

            # Custom per-module buffering — bypass initialize_online_processing.
            # vLLM's online processing (CopyCounter) doesn't reliably
            # complete FusedMoE modules on meta device. We track loaded
            # numel directly from each weight_loader call instead.
            buffer: list[tuple[str, tuple, dict]] = []
            loaded_numel = [0]
            materialized = [False]
            moe_packed = {
                "pending_packed": {},
                "pending_norms": {},
                "shard_order": {"w13_weight": [], "w2_weight": []},
                "device": None,
            }

            def _moe_dims(param_name: str) -> tuple[int, int]:
                from turboquant_vllm.weight_quant import packed_group_bytes, padded_size

                shape = param_shapes[param_name]
                in_dim = shape[-1]
                _, n_groups = padded_size(in_dim, self.group_size)
                bytes_per_group = packed_group_bytes(self.bits, self.group_size)
                return n_groups, bytes_per_group

            def _record_packed_moe(param_name: str, loaded_weight: torch.Tensor, args, kwargs) -> bool:
                if not isinstance(loaded_weight, torch.Tensor):
                    return False
                weight_name = _extract_weight_name(args, kwargs)
                is_packed = weight_name is not None and weight_name.endswith(".tq_packed")
                is_norms = weight_name is not None and weight_name.endswith(".tq_norms")
                n_groups, bytes_per_group = _moe_dims(param_name)
                if not is_packed and loaded_weight.dtype == torch.uint8 and loaded_weight.ndim == 2:
                    is_packed = loaded_weight.shape[1] == bytes_per_group
                if not is_norms and loaded_weight.is_floating_point() and loaded_weight.ndim == 2:
                    is_norms = loaded_weight.shape[1] == n_groups
                if not (is_packed or is_norms):
                    return False
                expert_id = kwargs.get("expert_id")
                if expert_id is None and len(args) >= 3 and isinstance(args[2], int):
                    expert_id = args[2]
                if expert_id is None:
                    return False
                shard_id = kwargs.get("shard_id")
                key = (param_name, shard_id, expert_id)
                order = moe_packed["shard_order"][param_name]
                if shard_id not in order:
                    order.append(shard_id)
                if is_packed:
                    moe_packed["pending_packed"][key] = loaded_weight
                if is_norms:
                    moe_packed["pending_norms"][key] = loaded_weight
                moe_packed["device"] = loaded_weight.device
                return True

            def _packed_complete(param_name: str) -> bool:
                shard_ids = moe_packed["shard_order"][param_name]
                if not shard_ids:
                    return False
                n_experts = param_shapes[param_name][0]
                for shard_id in shard_ids:
                    for expert_id in range(n_experts):
                        key = (param_name, shard_id, expert_id)
                        if key not in moe_packed["pending_packed"] or key not in moe_packed["pending_norms"]:
                            return False
                return True

            def _build_moe_compressed(param_name: str) -> "Compressed3D":
                from turboquant_vllm.weight_quant import Compressed3D

                shard_ids = _ordered_shard_keys(moe_packed["shard_order"][param_name])
                n_experts = param_shapes[param_name][0]
                packed_all = []
                norms_all = []
                for expert_id in range(n_experts):
                    packed_parts = [moe_packed["pending_packed"][(param_name, sid, expert_id)] for sid in shard_ids]
                    norms_parts = [moe_packed["pending_norms"][(param_name, sid, expert_id)] for sid in shard_ids]
                    packed_device = packed_parts[0].device
                    packed_dtype = packed_parts[0].dtype
                    packed_mismatch = [
                        (index, part.device, part.dtype)
                        for index, part in enumerate(packed_parts[1:], start=1)
                        if part.device != packed_device or part.dtype != packed_dtype
                    ]
                    if packed_mismatch:
                        raise RuntimeError(
                            f"TQ3 packed MoE: {param_name} packed shards have mixed device/dtype "
                            f"(expected {packed_device}/{packed_dtype}, mismatches={packed_mismatch})."
                        )
                    norms_device = norms_parts[0].device
                    norms_dtype = norms_parts[0].dtype
                    norms_mismatch = [
                        (index, part.device, part.dtype)
                        for index, part in enumerate(norms_parts[1:], start=1)
                        if part.device != norms_device or part.dtype != norms_dtype
                    ]
                    if norms_mismatch:
                        raise RuntimeError(
                            f"TQ3 packed MoE: {param_name} norms shards have mixed device/dtype "
                            f"(expected {norms_device}/{norms_dtype}, mismatches={norms_mismatch})."
                        )
                    packed_all.append(torch.cat(packed_parts, dim=0))
                    norms_all.append(torch.cat(norms_parts, dim=0))
                packed = torch.cat(packed_all, dim=0)
                norms = torch.cat(norms_all, dim=0)
                return Compressed3D.from_packed(
                    packed,
                    norms,
                    shape=tuple(param_shapes[param_name]),
                    dtype=param_dtypes[param_name],
                    bits=self.bits,
                    group_size=self.group_size,
                )

            def _make_buffering_loader(param_name, orig_loader):
                def _buffering_loader(*args, **kwargs):
                    if materialized[0]:
                        return orig_loader(*args, **kwargs)
                    loaded_weight = args[1] if len(args) > 1 else None
                    if isinstance(loaded_weight, torch.Tensor) and _record_packed_moe(
                        param_name, loaded_weight, args, kwargs
                    ):
                        if _packed_complete("w13_weight") and _packed_complete("w2_weight"):
                            materialized[0] = True
                            layer._tq_w13_weight = _build_moe_compressed("w13_weight")
                            layer._tq_w2_weight = _build_moe_compressed("w2_weight")
                            device = moe_packed["device"] or layer._tq_w13_weight.packed.device
                            _materialize_packed_moe(
                                layer,
                                orig_loaders,
                                param_shapes,
                                param_dtypes,
                                device,
                                {"w13_weight", "w2_weight"},
                            )
                            moe_packed["pending_packed"].clear()
                            moe_packed["pending_norms"].clear()
                            self._do_compress(layer)
                        return True
                    numel = loaded_weight.numel() if isinstance(loaded_weight, torch.Tensor) else 0
                    buffer.append((param_name, args, kwargs))
                    loaded_numel[0] += numel
                    if loaded_numel[0] >= total_numel:
                        materialized[0] = True
                        _materialize_and_process(
                            layer,
                            buffer,
                            orig_loaders,
                            param_shapes,
                            param_dtypes,
                            self,
                        )
                    # Signal success so model.load_weights commits the expert
                    return True

                return _buffering_loader

            for pname, param in layer.named_parameters(recurse=False):
                if pname in orig_loaders:
                    param.weight_loader = _make_buffering_loader(
                        pname,
                        orig_loaders[pname],
                    )

        def _do_compress(self, layer: nn.Module) -> None:
            """Kernel setup + TQ3 compression. Called after materialization."""
            global _shared_moe_scratch_pool

            from turboquant_vllm.moe_quant import TurboQuantFusedMoEScratchPool
            from turboquant_vllm.weight_quant import _compress_3d_param

            prepacked = hasattr(layer, "_tq_w13_weight") and hasattr(layer, "_tq_w2_weight")

            w13 = getattr(layer, "w13_weight", None)
            w2 = getattr(layer, "w2_weight", None)
            if w13 is None or w2 is None or w13.dim() != 3 or w2.dim() != 3:
                return

            if not prepacked:
                self._unquant.process_weights_after_loading(layer)
                _compress_3d_param(layer, "w13_weight", self.bits, self.group_size)
                _compress_3d_param(layer, "w2_weight", self.bits, self.group_size)

            self._w13_c = layer._tq_w13_weight
            self._w2_c = layer._tq_w2_weight

            if _shared_moe_scratch_pool is None:
                _shared_moe_scratch_pool = TurboQuantFusedMoEScratchPool(
                    self._w13_c,
                    self._w2_c,
                )
            else:
                _shared_moe_scratch_pool.assert_matches(
                    self._w13_c,
                    self._w2_c,
                )

            self._pool = _shared_moe_scratch_pool
            layer.w13_weight.data = self._pool.w13
            layer.w2_weight.data = self._pool.w2

            if prepacked:
                self._unquant.process_weights_after_loading(layer)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            # Compression handled by _materialize_and_process (triggered
            # by buffering loader). This guard handles the global sweep.
            if not hasattr(layer, "_tq_w13_weight"):
                # Not yet compressed — run compression now (fallback
                # for modules where buffering didn't trigger)
                if hasattr(layer, "w13_weight") and layer.w13_weight.numel() > 0:
                    self._do_compress(layer)

        def get_fused_moe_quant_config(self, layer: nn.Module):
            return self._unquant.get_fused_moe_quant_config(layer)

        def apply(self, layer: nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
            # Decompress into shared scratch pool, then delegate to
            # the unquantized method which has the MoE kernel.
            if self._pool is not None and self._w13_c is not None:
                self._w13_c.decompress_into(
                    self._pool.w13,
                    fp32_scratch=self._pool.w13_fp32,
                )
                self._w2_c.decompress_into(
                    self._pool.w2,
                    fp32_scratch=self._pool.w2_fp32,
                )
            return self._unquant.apply(layer, x, **kwargs)

else:
    TurboQuantOnlineMoEMethod = None  # type: ignore[assignment,misc]


_registered = False


def register():
    """Register TurboQuant as a vLLM quantization method. Called from the plugin."""
    global _registered
    if _registered:
        return
    _registered = True

    if LinearBase is None:
        logger.debug("vLLM not installed, skipping TurboQuant quant config registration")
        return

    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config("turboquant")(TurboQuantConfig)
    _patch_weight_name_remapping()
    logger.info("TurboQuant quantization config registered with vLLM")


# FP8 metadata that survives a re-quantization to TQ3 as dead bytes.
_FP8_LEFTOVER_SCALE_SUFFIXES = (
    ".weight_scale_inv",
    ".weight_scale",
    ".input_scale",
)


def _patch_weight_name_remapping():
    """Monkey-patch vLLM's weight iterator to pass through TQ3 packed weights.

    As each ``.tq_packed`` / ``.tq_norms`` tensor arrives from the
    checkpoint iterator, yield it directly so TurboQuant's quant methods
    can bind packed buffers on GPU without bf16 decompression.
    """
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError:
        return

    _original_get_all_weights = DefaultModelLoader.get_all_weights

    def _decompress_get_all_weights(self, model_config, model):
        """Pass through ``.tq_packed`` + ``.tq_norms`` tensors as-is."""
        import os as _os

        tq_config_path = _os.path.join(model_config.model, "tq_config.json")
        if not _os.path.isfile(tq_config_path):
            try:
                from huggingface_hub import hf_hub_download

                revision = getattr(model_config, "revision", None)
                tq_config_path = hf_hub_download(
                    model_config.model,
                    "tq_config.json",
                    revision=revision,
                )
            except Exception as e:
                logger.info(
                    "No tq_config.json for %s (%s), passing through",
                    model_config.model,
                    e,
                )
                yield from _original_get_all_weights(self, model_config, model)
                return

        import json as _json

        with open(tq_config_path) as f:
            tq_cfg = _json.load(f)
        if tq_cfg.get("format") != "tq3_native":
            logger.info(
                "tq_config.json format %s is not tq3_native; using default loader",
                tq_cfg.get("format"),
            )
            yield from _original_get_all_weights(self, model_config, model)
            return
        bits = tq_cfg.get("bits", 3)
        group_size = tq_cfg.get("group_size", 128)
        logger.info(
            "TQ3 native checkpoint (bits=%d, group_size=%d): direct packed load",
            bits,
            group_size,
        )

        pending_packed: set[str] = set()
        pending_norms: set[str] = set()
        skipped_fp8_scales = 0

        for name, tensor in _original_get_all_weights(self, model_config, model):
            if name.endswith(".tq_packed"):
                base = name[: -len(".tq_packed")]
                if base in pending_norms:
                    pending_norms.remove(base)
                else:
                    pending_packed.add(base)
                yield name, tensor
            elif name.endswith(".tq_norms"):
                base = name[: -len(".tq_norms")]
                if base in pending_packed:
                    pending_packed.remove(base)
                else:
                    pending_norms.add(base)
                yield name, tensor
            elif name.endswith(_FP8_LEFTOVER_SCALE_SUFFIXES):
                skipped_fp8_scales += 1
                continue
            else:
                yield name, tensor

        if pending_packed:
            for base in sorted(pending_packed):
                logger.warning("Orphaned .tq_packed without .tq_norms: %s", base)
        if pending_norms:
            for base in sorted(pending_norms):
                logger.warning("Orphaned .tq_norms without .tq_packed: %s", base)
        if skipped_fp8_scales > 0:
            logger.info(
                "TQ3 native: dropped %d FP8 leftover scale tensors",
                skipped_fp8_scales,
            )

    DefaultModelLoader.get_all_weights = _decompress_get_all_weights
    logger.info("TQ3 packed-load hook installed on DefaultModelLoader.get_all_weights")
