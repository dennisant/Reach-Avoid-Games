from experiment.one_player_intersection_reachAvoid_timeConsistent import one_player_time_consistent
from experiment.one_player_intersection_reachAvoid_timeInconsistent import one_player_time_inconsistent
from utils.argument import *

args = get_argument()
check_argument(args)

try:
    if args.no_players == 1 and not args.time_consistency and args.env_type == "goal_with_obs":
        one_player_time_consistent(args)
    elif args.no_players == 1 and args.time_consistency and args.env_type == "goal_with_obs":
        one_player_time_inconsistent(args)

except:
    print("It seems like something is wrong, or the experiment you are trying to run is not available.")
    raise NotImplementedError