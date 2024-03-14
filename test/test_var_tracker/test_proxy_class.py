import torch
import json
class Proxy:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        # Intercept attribute access
        print(f"Accessing attribute '{name}'")
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name == '_obj':
            self.__dict__[name] = value  # Set the attribute directly
        else:
            # Intercept attribute assignment
            print(f"Setting attribute '{name}' to '{value}'")
            setattr(self._obj, name, value)
    
    def __delattr__(self, name):
        # Intercept attribute deletion
        print(f"Deleting attribute '{name}'")
        delattr(self._obj, name)
        
    def __getitem__(self, key):
        # Intercept item retrieval
        print(f"Getting item with key '{key}'")
        return self._obj[key]

    def __setitem__(self, key, value):
        # Intercept item assignment
        print(f"Setting item with key '{key}' to '{value}'")
        self._obj[key] = value
        
    def __delitem__(self, key):
        # Intercept item deletion
        print(f"Deleting item with key '{key}'")
        del self._obj[key]


# Replace the module's __dict__ attribute with the Proxy class
# torch.__dict__ = Proxy(torch.__dict__)
json.__dict__ = Proxy(json.__dict__)

# Example usage
# Accessing attributes through the proxy
print(torch.tensor)  # This will print: Accessing attribute 'tensor'
# Modifying attributes through the proxy
torch.tensor = lambda x: print(f"Creating tensor with value '{x}'")  # This will print: Setting attribute 'tensor' to '<lambda>'
# Creating a tensor through the proxy
torch.tensor([1, 2, 3])  # This will print: Creating tensor with value '[1, 2, 3]'
# Deleting attributes through the proxy
del torch.tensor  # This will print: Deleting attribute 'tensor'

# Accessing attributes through the proxy
print(torch.__dict__['__name__'])  # This will print: Accessing attribute '__name__'
# Modifying attributes through the proxy
torch.__dict__['__name__'] = 'modified_torch'  # This will print: Setting attribute '__name__' to 'modified_torch'
# Creating a tensor through the proxy
print(torch.__name__)  # This will print: modified_torch
# Deleting attributes through the proxy
del torch.__dict__['__name__']  # This will print: Deleting attribute '__name__'


# Example usage
my_list = [1, 2, 3]

# Creating a proxy around the list
list_proxy = Proxy(my_list)

# Accessing items through the proxy
print(list_proxy[0])  # This will print: Getting item with key '0'

# Modifying items through the proxy
list_proxy[0] = 4    # This will print: Setting item with key '0' to '4'

# Extending the list through the proxy
list_proxy.extend([5, 6, 7])  # This will print: Setting item with key '3' to '[5, 6, 7]'

# Deleting items through the proxy
del list_proxy[0]    # This will print: Deleting item with key '0'



# Creating a dictionary
my_dict = {'a': 1, 'b': 2}

# Creating a proxy around the dictionary
dict_proxy = Proxy(my_dict)

# Accessing items through the proxy
print(dict_proxy['a'])  # This will print: Getting item with key 'a'

# Modifying items through the proxy
dict_proxy['a'] = 3    # This will print: Setting item with key 'a' to '3'

# Deleting items through the proxy
del dict_proxy['a']    # This will print: Deleting item with key 'a'


# Creating a tensor
my_tensor = torch.tensor([1, 2, 3])

# Creating a proxy around the tensor
tensor_proxy = Proxy(my_tensor)

# Accessing items through the proxy
print(tensor_proxy[0])  # This will print: Getting item with key '0'

# Modifying items through the proxy
tensor_proxy[0] = 4    # This will print: Setting item with key '0' to '4'

