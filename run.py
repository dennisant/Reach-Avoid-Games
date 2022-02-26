from experiment.one_player_goal_with_obs import one_player
from experiment.three_player_t_intersection import three_player
from experiment.two_player_t_intersection import two_player
from experiment.two_player_t_intersection_adversarial import two_player_adversarial
from utils.argument import *

args = get_argument()
check_argument(args)

note = input("Do you want to note something about this run?: ")
args.note = note

try:
    if args.no_players == 1 and args.env_type == "goal_with_obs":
        one_player(args)
    elif args.no_players == 2 and args.env_type == "t_intersection" and args.t_react is None:
        two_player(args)
    elif args.no_players == 2 and args.env_type == "t_intersection" and args.t_react is not None:
        two_player_adversarial(args)
    elif args.no_players == 3 and args.env_type == "t_intersection":
        three_player(args)
    else:
        raise NotImplementedError("There is no implementation currently available for your choice.")

except:
    raise NotImplementedError("It seems like something is wrong, or the experiment you are trying to run is not available.")