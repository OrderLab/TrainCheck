from mldaikon.instrumentor import meta_vars


def annotate_stage(stage_name: str):
    """Annotate the current stage. This function should be invoked as the very first statement of the stage.
    A stage is invalidated after a new stage annotation is encountered.

    Allowed stage names: `init`, `training`, `evaluation`, `inference`, `testing`, `checkpointing`, `preprocessing`, `postprocessing`

    Note that it is your responsibility to make sure this function is called on all threads that potentially can generate invariant candidates.
    """

    valid_stage_names = [
        "init",
        "training",
        "evaluation",
        "inference",
        "testing",
        "checkpointing",
        "preprocessing",
        "postprocessing",
    ]

    assert (
        stage_name in valid_stage_names
    ), f"Invalid stage name: {stage_name}, valid ones are {valid_stage_names}"

    meta_vars["stage"] = stage_name
