from experiment.one_player_intersection_reachAvoid_timeConsistent import one_player_time_consistent
from experiment.one_player_intersection_reachAvoid_timeInconsistent import one_player_time_inconsistent
from experiment.three_player_intersection_reachAvoid_timeConsistent import three_player_time_consistent
from experiment.three_player_intersection_reachAvoid_timeInconsistent import three_player_time_inconsistent
from utils.argument import *

args = get_argument()
check_argument(args)

try:
    if args.no_players == 1 and not args.time_consistency and args.env_type == "goal_with_obs":
        one_player_time_inconsistent(args)
    elif args.no_players == 1 and args.time_consistency and args.env_type == "goal_with_obs":
        one_player_time_consistent(args)
    elif args.no_players == 2 and not args.time_consistency and args.env_type == "t_intersection":
        pass
    elif args.no_players == 2 and args.time_consistency and args.env_type == "t_intersection":
        pass
    elif args.no_players == 2 and not args.time_consistency and args.env_type == "t_intersection" and args.adversarial:
        pass
    elif args.no_players == 2 and args.time_consistency and args.env_type == "t_intersection" and args.adversarial:
        pass
    elif args.no_players == 3 and not args.time_consistency and args.env_type == "t_intersection":
        three_player_time_inconsistent(args)
    elif args.no_players == 3 and args.time_consistency and args.env_type == "t_intersection":
        three_player_time_consistent(args)
    else:
        raise NotImplementedError("There is no implementation currently available for your choice.")

except:
    raise NotImplementedError("It seems like something is wrong, or the experiment you are trying to run is not available.")