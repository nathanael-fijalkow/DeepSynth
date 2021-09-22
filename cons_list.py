# list = None | (value, list)

def index(cons_list, i):
    try:
        (value, next_const_list) = cons_list
        if i == 0: 
            return value
        else:
            return index(next_const_list, i-1)
    except:
        print(f"cons_list.py: cons_list is empty at i={i}!")
        return None

def tuple2constlist(t, i = 0):
    if i < len(t):
        return (t[i], tuple2constlist(t, i+1))
    else:
        return None


def length(cons_list):
    try:
        _, queue = cons_list
        return 1 + length(queue)
    except:
        return 0


def cons_list2list(cons_list, l = None):
    l = l or []
    try:
        x, queue = cons_list
        l.append(x)
        return cons_list2list(queue, l)
    except:
        return l


def cons_list_split(cons_list, n):
    if n == 0:
        return None, cons_list
    head, tail = cons_list
    correct_length_part, rest = cons_list_split(tail, n - 1)
    return (head, correct_length_part), rest
