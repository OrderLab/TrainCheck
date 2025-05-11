import traincheck.instrumentor.tracer as tracer
from traincheck.config.config import ALL_STAGE_NAMES
from traincheck.instrumentor import meta_vars


def annotate_stage(stage_name: str):
    """Annotate the current stage. This function should be invoked as the very first statement of the stage.
    A stage is invalidated after a new stage annotation is encountered.

    Allowed stage names: `init`, `training`, `evaluation`, `inference`, `testing`, `checkpointing`, `preprocessing`, `postprocessing`

    Note that it is your responsibility to make sure this function is called on all threads that potentially can generate invariant candidates.
    """

    assert (
        stage_name in ALL_STAGE_NAMES
    ), f"Invalid stage name: {stage_name}, valid ones are {ALL_STAGE_NAMES}"

    meta_vars["stage"] = stage_name


def annotate_answer_start_token_ids(
    answer_start_token_id: int, include_start_token: bool = False
):
    """Annotate the answer start token ids specifically for prompt generation tasks. If this is provided, ML-DAIKON will look for transformer.generate method calls
    and use this information to only dump the response tokens.

    Args:
        answer_start_token_ids (list[int]): List of token ids that correspond to the start of the answer in the input sequence.
        include_start_tokens (bool): Whether to include the start tokens in the generated prompt. Default is False.

    Note that if the start tokens are not found in the generated prompt, ML-DAIKON will assume that the response length is zero and dump an empty response.

    If the answer_start_token_ids corresponds to the end of the input prompt, set include_start_tokens to False,
    Otherwise the token_ids should correspond to the start of the answer in the input prompt, and include_start_tokens should be set to True.
    """
    tracer.GENERATE_START_TOKEN_ID = answer_start_token_id
    tracer.GENERATE_START_TOKEN_ID_INCLUDE_START_TOKEN = include_start_token
