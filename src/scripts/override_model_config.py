import os

from torchtitan.tools.logging import logger

from src.codex import codex_configs


def _override_model_config_from_env() -> None:

    flavor = os.getenv("CODEX_FLAVOR", "small")
    cfg = codex_configs.get(flavor)
    if cfg is None:
        raise RuntimeError(
            f"codex_configs['{flavor}'] is missing; cannot override model config."
        )

    width_str = os.getenv("CODEX_WIDTH")
    depth_str = os.getenv("CODEX_DEPTH")
    if not (width_str or depth_str):
        logger.warning(
            "Neither CODEX_WIDTH nor CODEX_DEPTH is set; keeping existing Codex model configuration."
        )
        return

    if width_str and depth_str:
        raise RuntimeError("Both CODEX_WIDTH and CODEX_DEPTH cannot be set.")

    if width_str:
        try:
            width = int(width_str)
            logger.info(f"Overriding Codex '{flavor}' width to {width}.")
            cfg.d_model = width
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid CODEX_WIDTH value '{width_str}', expected integer."
            ) from exc

    if depth_str:
        try:
            depth = int(depth_str)
            logger.info(f"Overriding Codex '{flavor}' n_layers to {depth}.")
            cfg.n_layers = depth
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid CODEX_DEPTH value '{depth_str}', expected integer."
            ) from exc

    # Recalculate dependent dimensions to keep the config consistent.
    cfg._apply_dynamic_dims()


_override_model_config_from_env()
