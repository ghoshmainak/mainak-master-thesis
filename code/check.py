import pprint


def contract_file():
    contract_dict = {}
    with open("contract.txt", "r") as contract:
        for line in contract.readlines():
            tmp = line.strip("\n").split(",")
            contract_dict[tmp[0]] = tmp[1]
    pprint.pprint(contract_dict)


def check_dim(filename):
    with open(filename, 'r') as fin:
        first_line = fin.readline()
        embedding = first_line.split()[1:]
        return len(embedding)

