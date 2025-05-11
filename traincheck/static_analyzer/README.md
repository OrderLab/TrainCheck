# Static Analyzer

Note: This module (currently) mainly support dynamic graph unparsing from Pyan results and obtain function level info for proxy_class observer and (possibly) API instrumentor. Currently log is generated on torch2.2.2

# Requirements

Install necessary packages when using static analyzer, as we need to look into the specfic file.

# Usage

Edit the `config.py` or add compile flags to modify the behavior of the static analyzer. 

## To run the internal pytorch library: 
```bash
# Output all functions in torch.nn
python -m traincheck.static_analyzer.graph_generator.main --lib nn
# Or equivalently
python -m traincheck.static_analyzer.graph_generator.main --lib torch.nn

# Only output functions with namespace torch.nn.modules.padding
python -m traincheck.static_analyzer.graph_generator.main --lib nn --namespace torch.nn.modules.padding

# Only output functions with namespace torch.nn.modules.padding and used in torch.nn.modules.padding
python -m traincheck.static_analyzer.graph_generator.main --lib nn --namespace torch.nn.modules.padding --function torch.nn.modules.padding.ConstantPad3d
```

## To run the external library:
```bash
python -m traincheck.static_analyzer.graph_generator.main --ext ../../../example_pipelines/bug_84911_traincheck.py
```

# Thoughts

1. Because there is few classmethods/staticmethods, we are not collecting them. 

2. The roadmap is that we first look into low-level API (e.g., in torch.nn), and calculate the function call level using the top-down method. We found that filtering out layers greater than 3 is a good choice. However, there are many noises in the third layer. To filter out these noises, we look into higher level calls (e.g., `efficientnet_pytorch`) to see which is used. These are the functions we care about in the third layer.

3. The plan is to start from the high-level python script and parse modules used in the user scripts. We will do it recursively in the parsing.

4. Make a white (black) list to filter out some modules in the higher level scripts (e.g., modules that are not in torch, or efficientnet_pytorch, torchvision).

# TODOs

- [x] Add extern lib CLI.

- [x] Add a CLI to dump the Attributes.

- [ ] Add CLI to let user to specify how to filter the attributes.

- [x] Black list for pyan inputs (filenames)

- [x] Make one call graph for each of the modules in the user code

- [x] Only output non-hidden functions to the ".log" in way2

# Not urgent

- [ ] In 84911, line 124: `for name, param in model_transfer.module.named_parameters():`. The `named_parameters()` function is not captured. 

- [ ] If we want to get `torch.random_seed`, get the first level python files in the torch. 

- [ ] Function argument has a function call. It is marked as ^^^argument^^^
