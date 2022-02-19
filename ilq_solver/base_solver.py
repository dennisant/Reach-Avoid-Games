from abc import ABC, abstractmethod
import numpy as np
from solve_lq_game.solve_lq_game import solve_lq_game
import matplotlib.pyplot as plt

class BaseSolver(ABC):
    def __init__(self,
                 dynamics,
                 player_costs,
                 x0,
                 Ps,
                 alphas,
                 alpha_scaling= 1.0, # 0.01,
                 reference_deviation_weight=None,
                 logger=None,
                 visualizer=None,
                 u_constraints=None,
                 config=None):
        self._dynamics = dynamics
        #self._player_costs = player_costs
        self._x0 = x0
        self._Ps = Ps
        self._ns = None
        self._alphas = alphas
        self._u_constraints = u_constraints
        self._horizon = len(Ps[0])
        self._num_players = len(player_costs)
        self.exp_info = config["experiment"]
        self.g_params = config["g_params"]
        self.l_params = config["l_params"]
        self.time_consistency = config["args"].time_consistency
        self.max_steps = config["args"].max_steps
        self.is_batch_run = config["args"].batch_run

        self.plot = config["args"].plot
        self.log = config["args"].log
        self.vel_plot = config["args"].vel_plot
        self.ctl_plot = config["args"].ctl_plot
        self.hallucinated = config["args"].hallucinated
    
        self.config = config["args"]
        self._player_costs = player_costs

        # Current and previous operating points (states/controls) for use
        # in checking convergence.
        self._last_operating_point = None
        self._current_operating_point = None

        # Fixed step size for the linesearch.
        self._alpha_scaling = alpha_scaling

        # Reference deviation cost weight.
        self._reference_deviation_weight = reference_deviation_weight

        # Set up visualizer.
        self._visualizer = visualizer
        self._logger = logger

        self.linesearch = config["args"].linesearch
        self.linesearch_type = config["args"].linesearch_type

        # Log some of the paramters.
        if self._logger is not None and self.log:
            self._logger.log("horizon", self._horizon)
            self._logger.log("x0", self._x0)
            self._logger.log("config", self.config)
            self._logger.log("g_params", self.g_params)
            self._logger.log("l_params", self.l_params)
            self._logger.log("exp_info", self.exp_info)

        if self.linesearch and self.linesearch_type == "trust_region":
            self.margin = 5.0

    @abstractmethod
    def _TimeStar(self, xs, us, player_index, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _rollout(self, xs, us, player_index, **kwargs):
        raise NotImplementedError

    def _is_converged(self):
        if self._last_operating_point is None:
            return False

        if not self.time_consistency:
            if np.any(np.array(self._total_costs) > 0.0):
                return False
        else:
            if max([max(c).detach().numpy().flatten()[0] for c in self._costs]) > 0.0:
                return False

        return True

    def run(self):
        """ Run the algorithm for the specified parameters. """
        self.iteration = 0
        # Trying to store stuff in order to plot cost
        store_total_cost = []
        iteration_store = []
        store_freq = 5
        
        while not self._is_converged() and (self.max_steps is not None and self.iteration < self.max_steps):
            # # (1) Compute current operating point and update last one.
            xs, us = self._compute_operating_point()
            self._last_operating_point = self._current_operating_point
            self._current_operating_point = (xs, us)
            
            # (2) Linearize about this operating point. Make sure to
            # stack appropriately since we will concatenate state vectors
            # but not control vectors, so that
            #    ``` x_{k+1} - xs_k = A_k (x_k - xs_k) +
            #          sum_i Bi_k (ui_k - uis_k) ```
            As = []
            Bs = [[] for ii in range(self._num_players)]
            for k in range(self._horizon):
                A, B = self._dynamics.linearize_discrete(
                    xs[k], [uis[k] for uis in us])
                As.append(A)

                for ii in range(self._num_players):
                    Bs[ii].append(B[ii])
                    
            # (5) Quadraticize costs.
            # Get the hessians and gradients. Hess_x (Q) and grad_x (l) are zero besides at t*
            # Hess_u (R) and grad_u (r) are eps * I and eps * u_t, respectfully, for all [0, T]
            Qs = []
            ls = []
            rs = []
            costs = []
            Rs = []
            calc_deriv_cost = []
            value_func_plus = []
            func_array = []
            func_return_array = []
            total_costs = []
            first_t_stars = []
                        
            for ii in range(self._num_players):           
                Q, l, R, r, costss, total_costss, calc_deriv_cost_, func_array_, func_return_array_, value_func_plus_, first_t_star_ = self._TimeStar(xs, us, ii, first_t_star = True)

                Qs.append(Q[ii])
                ls.append(l[ii])
                rs.append(r[ii])
                
                costs.append(costss)
                calc_deriv_cost.append(calc_deriv_cost_)
                value_func_plus.append(value_func_plus_)
                func_array.append(func_array_)
                func_return_array.append(func_return_array_)
                total_costs.append(total_costss)
                first_t_stars.append(first_t_star_)
                
                Rs.append(R[ii])

            self._first_t_stars = first_t_stars
            self._Qs = Qs
            self._ls = ls
            self._rs = rs
            self._xs = xs
            self._us = us
            self._func_array = func_array
            self._func_return_array = func_return_array
            self._value_func_plus = value_func_plus
            self._calc_deriv_cost = calc_deriv_cost

            # Visualization.
            self.visualize()

            # (6) Compute feedback Nash equilibrium of the resulting LQ game.
            # This is getting put into compute_operating_point to solver
            # for the next trajectory
            # print(np.array(Qs).shape)
            # input()
            Ps, alphas, ns = solve_lq_game(As, Bs, Qs, ls, Rs, rs, calc_deriv_cost, self.time_consistency)

            # (7) Accumulate total costs for all players.
            # This is the total cost for the trajectory we are on now
            #total_costs = [sum(costis).item() for costis in costs]
            
            prompt = "\rTotal cost for player:"
            for i in range(self._num_players):
                prompt += "\t{:.3f}".format(total_costs[i])
            print(prompt, end="")
            
            self._total_costs = total_costs
            self._costs = costs
            
            #Store total cost at each iteration and the iterations
            store_total_cost.append(total_costs)
            iteration_store.append(self.iteration)

            # Update the member variables.
            self._Ps = Ps
            self._alphas = alphas
            self._ns = ns
            
            if self.linesearch:
                if self.linesearch_type == "trust_region":
                    self._alpha_scaling = self._linesearch_trustregion_ratio(iteration = self.iteration, visualize_hallucination=self.hallucinated)
                elif self.linesearch_type == "armijo":
                    self._alpha_scaling = self._linesearch_armijo(iteration = self.iteration)
                else:
                    self._alpha_scaling = 0.05
            else:
                self._alpha_scaling = 1.0 / ((self.iteration + 1) * 0.5) ** 0.3
                if self._alpha_scaling < .2:
                    self._alpha_scaling = .2

            # Log everything.
            if self._logger is not None and self.log and self.iteration%store_freq == 0:
                self._logger.log("xs", xs)
                self._logger.log("us", us)
                self._logger.log("total_costs", total_costs)
                self._logger.log("alpha_scaling", self._alpha_scaling)
                self._logger.log("calc_deriv_cost", calc_deriv_cost)
                self._logger.log("value_func_plus", value_func_plus)
                self._logger.log("func_array", func_array)
                self._logger.log("func_return_array", func_return_array)
                self._logger.log("first_t_star", first_t_stars)
                self._logger.log("iteration", self.iteration)
                self._logger.dump()
            
            self.iteration += 1

        if self._is_converged():
            print("\nExperiment converged")
            plt.figure()
            plt.plot(iteration_store, store_total_cost, color='green', linestyle='dashed', linewidth = 2, marker='o', markerfacecolor='blue', markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Total cost')
            plt.title('Total Cost of Trajectory at Each Iteration')
            if self.plot:
                plt.savefig(self.exp_info["figure_dir"] +'total_cost_after_{}_steps.jpg'.format(self.iteration)) # Trying to save these plots
            if not self.is_batch_run:
                plt.show()
            
            if self._logger is not None and self.log:
                self._logger.log("xs", xs)
                self._logger.log("us", us)
                self._logger.log("total_costs", total_costs)
                self._logger.log("alpha_scaling", self._alpha_scaling)
                self._logger.log("calc_deriv_cost", calc_deriv_cost)
                self._logger.log("value_func_plus", value_func_plus)
                self._logger.log("func_array", func_array)
                self._logger.log("func_return_array", func_return_array)
                self._logger.log("first_t_star", first_t_stars)
                self._logger.log("iteration", self.iteration)
                self._logger.log("is_converged", True)
                self._logger.dump()
        else:
            if self._logger is not None:
                self._logger.log("is_converged", False)
                self._logger.dump()

    @abstractmethod
    def _linesearch_trustregion(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _linesearch_trustregion_naive(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _linesearch_armijo(self, **kwargs):
        raise NotImplementedError

    def _compute_operating_point(self):
        """
        Compute current operating point by propagating through dynamics.

        :return: states, controls for all players (list of lists), and
            costs over time (list of lists), i.e. (xs, us, costs)
        :rtype: [np.array], [[np.array]], [[torch.Tensor(1, 1)]]
        """
        xs = [self._x0]
        us = [[] for ii in range(self._num_players)]
        #costs = [[] for ii in range(self._num_players)]

        for k in range(self._horizon):
            # If this is our fist time through, we don't have a trajectory, so
            # set the state and controls at each time-step k to 0. Else, use state and
            # controls
            if self._current_operating_point is not None:
                current_x = self._current_operating_point[0][k]
                current_u = [self._current_operating_point[1][ii][k]
                              for ii in range(self._num_players)]
            else:
                current_x = np.zeros((self._dynamics._x_dim, 1))
                current_u = [np.zeros((ui_dim, 1))
                              for ui_dim in self._dynamics._u_dims]
            
            # This is Eqn. 7 in the ILQGames paper
            # This gets us the control at time-step k for the updated trajectory
            feedback = lambda x, u_ref, x_ref, P, alpha : \
                        u_ref - P @ (x - x_ref) - self._alpha_scaling * alpha
            u = [feedback(xs[k], current_u[ii], current_x,
                          self._Ps[ii][k], self._alphas[ii][k])
                  for ii in range(self._num_players)]

            # Append computed control (u) for the trajectory we're calculating to "us"
            for ii in range(self._num_players):
                us[ii].append(u[ii])

            # Use 4th order Runge-Kutta to propogate system to next time-step k+1
            xs.append(self._dynamics.integrate(xs[k], u))
            
        #print("self._aplha_scaling in compute_operating_point is: ", self._alpha_scaling)
        return xs, us

    def visualize(self, **kwargs):
        xs = self._current_operating_point[0]
        us = self._current_operating_point[0]
        func_array = self._func_array
        func_return_array = self._func_return_array
        value_func_plus = self._value_func_plus
        calc_deriv_cost = self._calc_deriv_cost

        # TODO: flag these values
        plot_critical_points = True
        plot_car_for_critical_points = False

        if self._visualizer is not None:
            traj = {"xs" : xs}
            for ii in range(self._num_players):
                traj["u%ds" % (ii + 1)] = us[ii]

            self._visualizer.add_trajectory(self.iteration, traj)
            if self.ctl_plot:
                self._visualizer.plot_controls(1)
                plt.pause(0.001)
                plt.clf()
                self._visualizer.plot_controls(2)
                plt.pause(0.001)
                plt.clf()
            self._visualizer.plot()

            if plot_critical_points:
                for i in range(self._num_players):
                    if not self.time_consistency:
                        pinch_point_index = calc_deriv_cost[i].index("True")
                    g_critical_index = np.where(np.array(func_array[i]) == "g_x")[0]
                    l_critical_index = np.where(np.array(func_array[i]) == "l_x")[0]

                    g_critical_index_pos = []
                    g_critical_index_neg = []
                    for index in g_critical_index:
                        if value_func_plus[i][index] >= 0:
                            g_critical_index_pos.append(index)
                        else:
                            g_critical_index_neg.append(index)
                    
                    l_critical_index_pos = []
                    l_critical_index_neg = []
                    for index in l_critical_index:
                        if value_func_plus[i][index] >= 0:
                            l_critical_index_pos.append(index)
                        else:
                            l_critical_index_neg.append(index)

                    self._visualizer.draw_real_car(i, np.array(xs)[[0]])
                    if plot_car_for_critical_points:
                        self._visualizer.draw_real_car(i, np.array(xs)[g_critical_index])
                        self._visualizer.draw_real_car(i, np.array(xs)[l_critical_index])
                    else:
                        plt.figure(1)
                        if not self.time_consistency:
                            if func_array[i][pinch_point_index] == "g_x":
                                plt.scatter(np.array(xs)[pinch_point_index, 5*i], np.array(xs)[pinch_point_index, 5*i + 1], color="r", s=40, marker="*", zorder=20)
                            else:
                                plt.scatter(np.array(xs)[pinch_point_index, 5*i], np.array(xs)[pinch_point_index, 5*i + 1], color="r", s=40, marker="o", zorder=20)
                        else:
                            plt.scatter(np.array(xs)[g_critical_index, 5*i], np.array(xs)[g_critical_index, 5*i + 1], color="k", s=40, marker="*", zorder=20)
                            plt.scatter(np.array(xs)[l_critical_index, 5*i], np.array(xs)[l_critical_index, 5*i + 1], color="magenta", s=20, marker="o", zorder=20)
                        
                        plt.scatter(np.array(xs)[g_critical_index_pos, 5*i], np.array(xs)[g_critical_index_pos, 5*i + 1], color="k", s=40, marker="*", zorder=10)
                        plt.scatter(np.array(xs)[l_critical_index_pos, 5*i], np.array(xs)[l_critical_index_pos, 5*i + 1], color="magenta", s=20, marker="o", zorder=10)
                        plt.scatter(np.array(xs)[g_critical_index_neg, 5*i], np.array(xs)[g_critical_index_neg, 5*i + 1], color="y", s=40, marker="*", zorder=10)
                        plt.scatter(np.array(xs)[l_critical_index_neg, 5*i], np.array(xs)[l_critical_index_neg, 5*i + 1], color="y", s=20, marker="o", zorder=10)
            
            plt.pause(0.001)
            if self.plot:
                plt.savefig(self.exp_info["figure_dir"] +'plot-{}.jpg'.format(self.iteration)) # Trying to save these plots
            plt.clf()

        # draw velocity and timestar overlay graph for 2 cars
        if self.vel_plot:
            for i in range(self._num_players):
                g_critical_index = np.where(np.array(func_array[i]) == "g_x")[0]
                l_critical_index = np.where(np.array(func_array[i]) == "l_x")[0]
                value_critical_index = np.where(np.array(func_array[i]) == "value")[0]
                gradient_critical_index = np.where(np.array(func_array[i]) != "value")[0]
                plt.figure(4+i)
                vel_array = np.array([x[5*i + 4] for x in xs]).flatten()
                
                plt.plot(vel_array)
                plt.scatter(g_critical_index, vel_array[g_critical_index], color = "r")
                plt.scatter(l_critical_index, vel_array[l_critical_index], color = "g")
                plt.scatter(value_critical_index, vel_array[value_critical_index], color = "y")

                name_list = []
                try:
                    for func in np.array(func_return_array[i])[gradient_critical_index]:
                        try:
                            name_list.append(func.__name__.replace("_",""))
                        except:
                            name_list.append(type(func).__name__.replace("_",""))
                    for j in range(len(gradient_critical_index)):
                        plt.text(gradient_critical_index[j], vel_array[gradient_critical_index][j], name_list[j])
                except Exception as err:
                    print(err)
                    pass
                plt.pause(0.01)
                plt.clf()