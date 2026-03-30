
def Nargout(*args):
    import traceback
    callInfo = traceback.extract_stack()
    callLine = str(callInfo[-3].line)
    split_equal = callLine.split("=")
    split_comma = split_equal[0].split(",")
    num = len(split_comma)
    return args[0:num] if num > 1 else args[0]