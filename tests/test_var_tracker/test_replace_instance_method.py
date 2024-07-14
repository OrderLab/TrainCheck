class MyClass:
    def original_method(self):
        print("Original Method")


def new_method(self):
    print("New Method")


# Replace 'original_method' in MyClass with 'new_method'
setattr(MyClass, "original_method", new_method)

# All future instances will use the new method
obj = MyClass()
obj.original_method()  # Prints "New Method"
