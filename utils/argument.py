import argparse

def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log",    help="Turn on log for exp",         action="store_true")
    parser.add_argument("--plot",   help="Turn on plot for exp",        action="store_true")
    parser.add_argument("--vel",    help="Turn on vel plot for exp",    action="store_true")
    
    parser.add_argument("--draw_cars",    help="Draw cars instead of points",    action="store_true")

    return parser.parse_args()