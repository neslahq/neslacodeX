import os

from torchtitan.tools.logging import logger

from src.codex import codex_configs


def _override_model_config_from_env() -> None:

    cfg = codex_configs.get("small")
    if cfg is None:
        raise RuntimeError(
            "codex_configs['small'] is missing; cannot override model config."
        )

    width_str = os.getenv("CODEX_D_MODEL")
    depth_str = os.getenv("CODEX_DEPTH")
    if not (width_str or depth_str):
        logger.warning(
            "Neither CODEX_D_MODEL nor CODEX_DEPTH is set; keeping existing Codex model configuration."
        )
        return

    if width_str and depth_str:
        raise RuntimeError("Both CODEX_D_MODEL and CODEX_DEPTH cannot be set.")

    if width_str:
        try:
            width = int(width_str)
            logger.info(f"Overriding Codex 'small' d_model to {width}.")
            cfg.d_model = width
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid CODEX_D_MODEL value '{width_str}', expected integer."
            ) from exc

    if depth_str:
        try:
            depth = int(depth_str)
            logger.info(f"Overriding Codex 'small' n_layers to {depth}.")
            cfg.n_layers = depth
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid CODEX_DEPTH value '{depth_str}', expected integer."
            ) from exc

    # Recalculate dependent dimensions to keep the config consistent.
    cfg._apply_dynamic_dims()


_override_model_config_from_env()
