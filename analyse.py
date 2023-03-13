import sys, getopt
from srcs.common import load_data, error
from srcs.describe import describe
from srcs.pair_plot import scatterplot

type_analyse = ["describe", "pairplot"]

header = ["ID",
    "Diagnosis",
    "Radius Mean",
    "Texture Mean",
    "Perimeter Mean",
    "Area Mean",
    "Smoothness Mean",
    "Compactness Mean",
    "Concavity Mean",
    "Concave Points Mean",
    "Symmetry Mean",
    "Fractal Dimension Mean",
    "Radius SE",
    "Texture SE",
    "Perimeter SE",
    "Area SE",
    "Smoothness SE",
    "Compactness SE",
    "Concavity SE",
    "Concave Points SE",
    "Symmetry SE",
    "Fractal Dimension SE",
    "Radius Worst",
    "Texture Worst",
    "Perimeter Worst",
    "Area Worst",
    "Smoothness Worst",
    "Compactness Worst",
    "Concavity Worst",
    "Concave Points Worst",
    "Symmetry Worst",
    "Fractal Dimension Worst",]

def usage(string = None):
    if string is not None:
        print(string)
    print("usage: analyse -file=DATA --type=[describe | pairplot] --begin=X --end=X")
    print("\t-f | --file=  : 'dataset.csv'")
    print("\t-t | --type=  : type of analyse (describe | pairplot) default: describe")
    print("\t-b | --begin= : first line of analysis (>1)")
    print("\t-b | --end=   : last line of analysis")
    exit(1)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:t:b:e:", ["file=", "type=","begin=", "end=", ])
    except getopt.GetoptError as inst:
        usage(inst)
    try:
        data = None
        type = "describe"
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                data = load_data(arg, header=None, names=header)
            if opt in ["-t", "--type"]:
                type = arg
                if type not in type_analyse:
                    usage()
        if data is None:
            usage()
        begin, end = 0, data.shape[1]
        for opt, arg in opts:
            if opt in ["-b", "--begin"]:
                begin = int(arg)
            elif opt in ["-e", "--end"]:
                end = int(arg)
        if type == "describe": 
            describe(data, begin, end, True)
        else:
            if begin == 0:
                begin = 1
            if end == data.shape[1]:
                end = data.shape[1] - 2
            if begin > 0 and end <= data.shape[1] - 2:
                scatterplot(data, begin+1, end)
            else:
                error(f"Begin must be > 0 and end <= {data.shape[1] - 2}")
    except Exception as inst:
        error(inst)
if __name__ == "__main__":
    main(sys.argv[1:])