import torch
class Proxy:
    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, new_obj):
        if new_obj != self._obj:
            print(f"obj has changed from {self._obj} to {new_obj}")
        self._obj = new_obj
        
obj = Proxy(torch.Tensor([10]))
obj.obj = 20  # This will print: obj has changed from 10 to 20
obj.obj = 30  # This will print: obj has changed from 20 to 30


model = Proxy(model)

._obj
.obj