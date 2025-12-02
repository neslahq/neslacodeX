import os

from torchtitan.tools.logging import logger

from . import codex_configs


def _override_width_from_env() -> None:
    width_str = os.getenv("CODEX_D_MODEL")
    if not width_str:
        logger.warning(
            "CODEX_D_MODEL is not set; keeping existing Codex width configuration."
        )
        return

    try:
        width = int(width_str)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid CODEX_D_MODEL value '{width_str}', expected integer."
        ) from exc

    cfg = codex_configs.get("small")
    if cfg is None:
        raise RuntimeError("codex_configs['small'] is missing; cannot override width.")

    logger.info(f"Overriding Codex 'small' d_model to {width}.")
    cfg.d_model = width
    # Recalculate dependent dimensions to keep the config consistent.
    cfg._apply_dynamic_dims()


_override_width_from_env()
