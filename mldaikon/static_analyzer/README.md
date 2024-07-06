# Static Analyzer

Note: This module (currently) mainly support dynamic graph unparsing from Pyan results and obtain function level info for proxy_class observer and (possibly) API instrumentor. Currently log is generated on torch2.2.2

# Usage

```bash
python main.py --lib nn

# Only output functions with namespace torch.nn.modules.padding
python main.py --lib nn --namespace torch.nn.modules.padding

# Only output functions with namespace torch.nn.modules.padding and used in torch.nn.modules.padding
python main.py --lib nn --namespace torch.nn.modules.padding --function torch.nn.modules.padding.ConstantPad3d
```
