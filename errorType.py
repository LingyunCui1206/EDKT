import pandas as pd
import re

def getErrorType():
    main_df = pd.read_csv("../Dataset/CodeWorkout/MainTable.csv")

    errorMessage = pd.unique(main_df["CompileMessageData"])
    print(len(errorMessage))

    pattern = r"line (\d+): (.+)"
    errors = {}

    for message in errorMessage:
        if not isinstance(message, str) or message == 'nan':
            continue
        match = re.match(pattern, message)
        if match:
            line_number = match.group(1)
            error_description = match.group(2)
            if line_number in errors:
                errors[line_number].append(error_description)
            else:
                errors[line_number] = [error_description]

    all_error_type = []
    for line, errors_list in errors.items():
        all_error_type.extend(errors_list)

    all_error_type = list(set(all_error_type))
    sorted_error_type = sorted(all_error_type)

    ptn_err1 = re.compile(r"error: variable .+ is already defined in method .+")
    err1 = [error for error in sorted_error_type if ptn_err1.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err1.match(error)]


    ptn_err2 = re.compile(r"error: variable .+ might not have been initialized")
    err2 = [error for error in sorted_error_type if ptn_err2.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err2.match(error)]


    ptn_err3 = re.compile(r"error: package .+ does not exist")
    err3 = [error for error in sorted_error_type if ptn_err3.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err3.match(error)]


    ptn_err4 = re.compile(r"error: no suitable method found for .+")
    err4 = [error for error in sorted_error_type if ptn_err4.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err4.match(error)]


    ptn_err5 = re.compile(r"error: method .+ in class .+ cannot be applied to given types")
    err5 = [error for error in sorted_error_type if ptn_err5.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err5.match(error)]


    ptn_err6 = re.compile(r"error: non-static method .+ cannot be referenced from a static context")
    err6 = [error for error in sorted_error_type if ptn_err6.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err6.match(error)]


    ptn_err7 = re.compile(r"error: integer number too large: .+")
    err7 = [error for error in sorted_error_type if ptn_err7.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err7.match(error)]


    ptn_err8 = re.compile(r"error: incompatible types: .+")
    err8 = [error for error in sorted_error_type if ptn_err8.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err8.match(error)]


    ptn_err9 = re.compile(r"error: incomparable types: .+")
    err9 = [error for error in sorted_error_type if ptn_err9.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err9.match(error)]


    ptn_err10 = re.compile(r"error: illegal .+")
    err10 = [error for error in sorted_error_type if ptn_err10.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err10.match(error)]


    ptn_err11 = re.compile(r"error: cannot find symbol: variable .+")
    err11 = [error for error in sorted_error_type if ptn_err11.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err11.match(error)]


    ptn_err12 = re.compile(r"error: cannot find symbol: method .+")
    err12 = [error for error in sorted_error_type if ptn_err12.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err12.match(error)]


    ptn_err13 = re.compile(r"error: cannot find symbol: class .+")
    err13 = [error for error in sorted_error_type if ptn_err13.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err13.match(error)]


    ptn_err14 = re.compile(r"error: bad operand type.+")
    err14 = [error for error in sorted_error_type if ptn_err14.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err14.match(error)]


    ptn_err15 = re.compile(r"error: array required.+")
    err15 = [error for error in sorted_error_type if ptn_err15.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err15.match(error)]


    ptn_err16 = re.compile(r"error: .+ expected")
    err16 = [error for error in sorted_error_type if ptn_err16.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err16.match(error)]

    
    ptn_err17 = re.compile(r"error: .+ character .+")
    err17 = [error for error in sorted_error_type if ptn_err17.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err17.match(error)]


    ptn_err18 = re.compile(r"error:.+?variable.+")
    err18 = [error for error in sorted_error_type if ptn_err18.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err18.match(error)]

    
    ptn_err19 = re.compile(r"error:.+?method .+")
    err19 = [error for error in sorted_error_type if ptn_err19.match(error)]
    sorted_error_type = [error for error in sorted_error_type if not ptn_err19.match(error)]


    # print("----- Updated sorted_error_type list:")
    # for error in sorted_error_type:
    #     print(error)


    ErrorType1 = err1
    
    ErrorType2 = err2
    
    ErrorType3 = err11+err18

    ErrorType4 = err7

    ErrorType5 = err15+err4+err6+err12+err19

    ErrorType6 = err13+err5

    ErrorType7 = err14

    ErrorType8 = err10+err16+err17

    ErrorType9 = err3

    ErrorType10 = err8+err9

    ErrorType11 = sorted_error_type
    
    ErrorType0 = []
    
    errors = [ErrorType0, ErrorType1, ErrorType2, ErrorType3, ErrorType4, ErrorType5, ErrorType6, ErrorType7, ErrorType8, ErrorType9, ErrorType10, ErrorType11]
    # for e in errors:
    #     print(len(e))
    #     print(e)
    #     print("------------")
    return errors








