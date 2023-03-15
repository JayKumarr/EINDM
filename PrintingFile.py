

def print_list(filepath, list, sep=","):
    f = open(filepath, "w")
    for item in list:
        st = str(item)+sep+" "
        f.write(st)
    f.close()

def print_dic(filepath, list, sep=",", new_item_sep = "\n"):
    f = open(filepath, "a")
    for item, value in list.items():
        st = str(item)+sep+" "+str(value) + new_item_sep
        f.write(st)
    f.close()