import sys, getopt
from srcs.common import load_data, error
from srcs.describe import describe
from srcs.pair_plot import scatterplot
from srcs.header import header

type_analyse = ["describe", "pairplot"]

def usage(string = None):
    if string is not None:
        print(string)
    print("usage: analyse --file=DATA --type=[describe | pairplot] --begin=X --end=X")
    print("\t-f | --file=  : 'dataset.csv'")
    print("\t-t | --type=  : type of analyse (describe | pairplot) default: describe")
    print("\t-b | --begin= : first column of analysis (>1)")
    print("\t-e | --end=   : last column of analysis")
    exit(1)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:t:b:e:h", ["file=", "type=","begin=", "end=", "help"])
    except getopt.GetoptError as inst:
        usage(inst)
    try:
        data = None
        type = "describe"
        begin, end = None, None
        for opt, arg in opts:
            if opt in ["-h", "--help"]:
                usage()
            elif opt in ["-f", "--file"]:
                data = load_data(arg, header=None, names=header)
            elif opt in ["-t", "--type"]:
                type = arg
                if type not in type_analyse:
                    usage()
            elif opt in ["-b", "--begin"]:
                begin = int(arg)
            elif opt in ["-e", "--end"]:
                end = int(arg)
        if data is None:
            usage()
        if begin is None:
            begin = 0
        if end is None:
            end = data.shape[1]
        if type == "describe": 
            describe(data, begin, end, True)
        else:
            if begin == 0:
                begin = 1
            if end == data.shape[1]:
                end = data.shape[1] - 2
            if begin >= 0 and end <= data.shape[1] - 2:
                scatterplot(data, begin, end)
            else:
                error(f"Begin must be > 0 and end <= {data.shape[1] - 2}")
    except Exception as inst:
        error(inst)
if __name__ == "__main__":
    main(sys.argv[1:])