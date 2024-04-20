from deepdiff import DeepDiff


def diffStates(state1, state2):
    # FIXME: convert all lists to string
    for key in state1:
        if isinstance(state1[key], list):
            state1[key] = str(state1[key])
    for key in state2:
        if isinstance(state2[key], list):
            state2[key] = str(state2[key])

    diff = DeepDiff(state1, state2)
    diff_properties = []
    for types in diff:
        for prop in diff[types]:
            prop = prop.split("['")[-1].split("']")[0]
            diff_properties.append(prop)
    return set(diff_properties)
