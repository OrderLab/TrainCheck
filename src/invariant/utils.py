from deepdiff import DeepDiff


def diffStates(state1, state2):
    diff = DeepDiff(state1, state2)
    diff_properties = []
    for types in diff:
        for prop in diff[types]:
            prop = prop.split("['")[-1].split("']")[0]
            diff_properties.append(prop)
    return set(diff_properties)
