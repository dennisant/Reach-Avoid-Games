from experiment.one_player_intersection_reachAvoid import one_player
from experiment.three_player_intersection_reachAvoid import three_player
from utils.argument import *

args = get_argument()
check_argument(args)

try:
    if args.no_players == 1 and args.env_type == "goal_with_obs":
        one_player(args)
    elif args.no_players == 2 and not args.time_consistency and args.env_type == "t_intersection":
        raise NotImplementedError("two-player case not yet refactored")
    elif args.no_players == 2 and args.time_consistency and args.env_type == "t_intersection":
        raise NotImplementedError("two-player case not yet refactored")
    elif args.no_players == 2 and not args.time_consistency and args.env_type == "t_intersection" and args.adversarial:
        raise NotImplementedError("two-player case not yet refactored")
    elif args.no_players == 2 and args.time_consistency and args.env_type == "t_intersection" and args.adversarial:
        raise NotImplementedError("two-player case not yet refactored")
    elif args.no_players == 3 and args.env_type == "t_intersection":
        three_player(args)
    else:
        raise NotImplementedError("There is no implementation currently available for your choice.")

except:
    raise NotImplementedError("It seems like something is wrong, or the experiment you are trying to run is not available.")