from bytecode import *
import torch
a=10
tensor = torch.tensor([1, 2, 3])
tensor_dict = torch.__dict__
def instr_monitor_var(func, varname):
    print_bc = [Instr('LOAD_GLOBAL', 'print'), Instr('LOAD_GLOBAL', varname),
                Instr('CALL_FUNCTION', 1), Instr('POP_TOP')]

    bytecodes = Bytecode.from_code(func.__code__)
    for i in reversed(range(len(bytecodes))):
        if bytecodes[i].name=='STORE_GLOBAL' and bytecodes[i].arg==varname:
            bytecodes[i:i]=print_bc

    func.__code__=bytecodes.to_code()

def test():


    global tensor_dict
    
    tensor_dict = torch.__dict__
    torch.new_attribute = "new value" # track tensor_dict
    torch.new_attribute = "new value2" # track tensor_dict
    
    # track tensor
    global tensor
    
    tensor = torch.tensor([1, 2, 3])
    
    tensor = torch.cat((tensor, torch.tensor([4])), dim=0)
    
    tensor = torch.tensor([1, 2, 3])

instr_monitor_var(test, 'tensor_dict')
instr_monitor_var(test, 'tensor')
test()

statevarobserver observedfunc 