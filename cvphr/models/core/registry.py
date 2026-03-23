def merge_model_dicts(*dicts, strict=True):
    """
    Registry + explicit merge + conflict detection.
    Merge multiple model dictionaries into a single dictionary.
    Args:
        *dicts: Multiple model dictionaries to merge.
        strict (bool): Whether to raise an error if duplicate model names are found.
    Returns:
        dict: A single merged dictionary containing all model classes and their corresponding keyword arguments.
    Raises:
        KeyError: If duplicate model names are found and strict is True.
    """
    merged = {}
    for d in dicts:
        for k, v in d.items():
            if strict and k in merged:
                raise KeyError(
                    f"Duplicate model name '{k}' "
                    f"found in multiple MODEL_CLASS_DICTs"
                )
            merged[k] = v
    return merged
