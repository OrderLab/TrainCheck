import torch


def training_loop():
    input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(3, 2)

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 1),
        torch.nn.Sigmoid(),
    )

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params)

    for i in range(6):
        optimizer.zero_grad()
        # Test that step behaves as expected (a no-op) when grads are set to None
        if i != 3:
            output = model(input)
            loss = output.sum()
            loss.backward()

        optimizer.step()
        print("step", optimizer.state[params[0]]["step"])


compiled_training_loop = torch._dynamo.optimize("eager", save_config=False)(
    training_loop
)

print("expected in eager:")
training_loop()

print("what actually happens after dynamo:")
compiled_training_loop()
