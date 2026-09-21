# src/salience/build_salience.py
def build_salience_generator(cfg):
    data_cfg = cfg.get("data", {})
    salience_cfg = cfg.get("salience", {})
    data_salience_cfg = data_cfg.get("salience_params", {})

    plugin_name = salience_cfg.get(
        "plugin",
        data_salience_cfg.get("plugin", "original"),
    )

    if plugin_name == "original":
        from src.salience.plugin import SalienceGenerator
    elif plugin_name == "normalized":
        from src.salience.plugin_normalized import SalienceGenerator
    else:
        raise ValueError(
            f"Unsupported salience plugin: {plugin_name}. "
            "Expected 'original' or 'normalized'."
        )

    kwargs = {
        "delta_theta": salience_cfg.get(
            "delta_theta",
            data_salience_cfg.get("delta_theta", 15),
        ),
        "delta_sigma": salience_cfg.get(
            "delta_sigma",
            data_salience_cfg.get("delta_sigma", 1.0),
        ),
        "K": salience_cfg.get(
            "K",
            data_salience_cfg.get("K", 5),
        ),
    }

    if plugin_name == "normalized":
        kwargs["alpha"] = salience_cfg.get(
            "alpha",
            data_salience_cfg.get("alpha", 0.5),
        )

    salience_gen = SalienceGenerator(**kwargs)
    return salience_gen, plugin_name, kwargs

