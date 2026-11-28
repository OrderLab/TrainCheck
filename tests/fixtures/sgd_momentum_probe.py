"""Train a small model with SGD momentum and record the optimizer state (issue #176).

Run plainly and under ``traincheck.collect_trace``; the two JSON reports must be
identical. Floats are recorded with ``float.hex`` so that equality is exact.
The report path is read from the ``TRAINCHECK_SGD_PROBE_OUT`` environment variable.
``traincheck_file`` names the traincheck installation that the run imported.
With ``TRAINCHECK_SGD_PROBE_WITHOUT_WRITEBACK=1`` the list write-back registry
is emptied first, as a control.
"""

import io
import json
import os
import sys

import torch


def hex_values(tensor: torch.Tensor) -> list[str]:
    return [float(value).hex() for value in tensor.detach().reshape(-1).tolist()]


def main() -> None:
    if os.environ.get("TRAINCHECK_SGD_PROBE_WITHOUT_WRITEBACK") == "1":
        # Control run under the collector: empty the list write-back registry
        # so that unproxied calls fall back to plain copying.
        from traincheck.instrumentor.proxy_wrapper import list_writeback

        dict.clear(list_writeback.LIST_WRITEBACK_ARGUMENTS)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 1)

    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()).hex())

    parameters = list(model.parameters())
    buffers: list[list[str] | None] = []
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        buffer = None if state is None else state.get("momentum_buffer")
        buffers.append(None if buffer is None else hex_values(buffer))

    report: dict[str, object] = {
        "losses": losses,
        "parameter_count": len(parameters),
        "state_count": len(optimizer.state),
        "state_keys_are_parameters": all(
            any(key is parameter for parameter in parameters) for key in optimizer.state
        ),
        "momentum_buffers": buffers,
        "parameters": [hex_values(parameter) for parameter in parameters],
    }

    checkpoint = io.BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    restored.load_state_dict(torch.load(checkpoint))
    restored.zero_grad()
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    restored.step()
    report["restored_loss"] = float(loss.item()).hex()
    report["restored_state_count"] = len(restored.state)
    report["restored_parameters"] = [
        hex_values(parameter) for parameter in model.parameters()
    ]

    # Under the collector, the instrumented copy of this script imports
    # traincheck; record which installation it used.
    traincheck = sys.modules.get("traincheck")
    report["traincheck_file"] = None if traincheck is None else traincheck.__file__

    with open(os.environ["TRAINCHECK_SGD_PROBE_OUT"], "w", encoding="utf-8") as file:
        json.dump(report, file)


if __name__ == "__main__":
    main()
