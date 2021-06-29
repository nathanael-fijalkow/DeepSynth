# list = None | (value, list)

def index(cons_list, i):
    try:
        (value, next_const_list) = cons_list
        if i == 0: 
            return value
        else:
            return index(next_const_list, i-1)
    except:
        print("Empty!")
        return None

def tuple2constlist(t, i = 0):
    if i < len(t):
        return (t[i], tuple2constlist(t, i+1))
    else:
        return None
