import argparse
from random import choices

def get_argument():
    parser = argparse.ArgumentParser()

    # experiment params
    parser.add_argument("--no_players",         help="Number of players",       type=int,       default=1)
    parser.add_argument("--time_consistency",   help="Is the run time consistent",  action="store_true")
    parser.add_argument("--adversarial",        help="Is the run adversarial",      action="store_true")

    parser.add_argument("--t_react",            help="T reaction for adversarial case",     type=int,       default=10)
    parser.add_argument("--t_horizon",          help="Time horizon for the traj",           type=float,     default=3.0)
    parser.add_argument("--t_resolution",       help="Time react",       type=float,     default=0.1)    

    parser.add_argument("--exp_name",           help="Name of experiment",              default="experiment")
    parser.add_argument("--player_types",       help="List of player types car/ped",    default=["car"])
    parser.add_argument("--init_states",        help="Init states for all players",     default=[0.0, 0.0, 0.0, 0.0, 0.0], type=float, nargs="*")

    parser.add_argument("--env_type",           help="Type of environment",             default=None,       choices=["goal_with_obs", "t_intersection"])
    # if goal_with_obs is chosen for env_type
    parser.add_argument("--obstacles",          help="List of obstacle in format [x, y, r]",    default=[6.0, 25.0, 4.0], type=float, nargs="*")
    parser.add_argument("--goal",               help="Goal information in format [x, y, r]",    default=[6.0, 40.0, 2.0], type=float, nargs="*")
    # if t_intersection is chosen for env_type
    # TODO

    # solver params
    parser.add_argument("--log",    help="Turn on log for exp",         action="store_true")
    parser.add_argument("--plot",   help="Turn on plot for exp",        action="store_true")
    parser.add_argument("--vel_plot",    help="Turn on vel plot for exp",    action="store_true")
    parser.add_argument("--ctl_plot",    help="Turn on ctl plot for exp",    action="store_true")
    
    # visualize params
    parser.add_argument("--draw_cars",    help="Draw cars instead of points",    action="store_true")
    parser.add_argument("--draw_roads",   help="Draw roads",            action="store_true")

    return parser.parse_args()

def check_argument(args):
    # Some logistic checking on the available experiments
    if args.no_players != 1 and args.adversarial:
        raise NotImplementedError("Experiment is not available, please choose another run.")

    # check information of env_type
    if args.env_type == "goal_with_obs":
        # check to make sure there is only one goal
        if (len(args.goal) % 3) != 0:
            raise TypeError("Something is wrong with your goal information, goal should be in the format of 'x y r'")
        elif int(len(args.goal) / 3) > 1:
            raise TypeError("Current implementation only supports single goal for this type of env")
        
        # check information of obstacles
        if (len(args.obstacles) % 3) != 0:
            raise TypeError("Something is wrong with your obs information, obs should be in the format of 'x y r'")

    elif args.env_type == "t_intersection":
        pass
    else:
        raise TypeError("You have not chosen any env_type to run")

    print("EXPERIMENT INFORMATION")
    print("\nGeneral information")
    print(" - Experiment name:\t\t\t {}".format(args.exp_name))
    print(" - Env type:\t\t\t\t {}".format(args.env_type))
    print(" - Horizon:\t\t\t\t {}".format(args.t_horizon))
    print(" - T resolution:\t\t\t {}".format(args.t_resolution))
    print(" - No. of players:\t\t\t {}".format(args.no_players))
    if args.env_type == "goal_with_obs":
        print(" - Obstacles:\t\t\t\t {}".format(args.obstacles))
        print(" - Goal:\t\t\t\t {}".format(args.goal))
        print(" - Init states:\t\t\t\t {}".format(args.init_states))
    elif args.env_type == "t_intersection":
        pass
    print("\nVisualization")
    print(" - Run with plots?:\t\t\t {}".format(args.plot))
    print(" - Run with control plot?:\t\t {}".format(args.ctl_plot))
    print(" - Run with velocity plot?:\t\t {}".format(args.vel_plot))
    print(" - Run with car figures?:\t\t {}".format(args.draw_cars))
    print(" - Run with graphical roads?:\t\t {}".format(args.draw_roads))
    print("\nLogging")
    print(" - Run with logs?:\t\t\t {}".format(args.log))