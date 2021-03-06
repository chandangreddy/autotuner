class Arguments:
    """Arguments from both the command line and the configuration file"""
    # Its attributes are filled up during command-line and configuration
    # file parsing

# Timing data
time_PPCG    = 0.0
time_backend = 0.0
time_binary  = 0.0

def summarise_timing():
    print("%s Summary of timing %s" % ('*' * 30, '*' * 30))
    print("Total time running PPCG:               %.2f seconds" % (time_PPCG))
    print("Total time running build:              %.2f seconds" % (time_backend))
    print("Total time running generated binaries: %.2f seconds" % (time_binary))
    print
