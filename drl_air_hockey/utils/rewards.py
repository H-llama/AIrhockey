import numpy as np


class TournamentReward:
    MAX_TIME_UNTIL_PENALTY_S: float = 15.0

    def __init__(
        self,
        reward_agent_score_goal: float = 1.0,
        reward_agent_receive_goal: float = -1.0,
        reward_opponent_faul: float = 1.0 / 3.0,
        reward_agent_faul: float = -1.0 / 3.0,
        reward_agent_cause_puck_stuck: float = 0.0,
    ):
        self._reward_agent_score_goal = reward_agent_score_goal
        self._reward_agent_receive_goal = reward_agent_receive_goal
        self._reward_opponent_faul = reward_opponent_faul
        self._reward_agent_faul = reward_agent_faul
        self._reward_agent_cause_puck_stuck = reward_agent_cause_puck_stuck
        self._penalty_threshold = 0.75 * self.MAX_TIME_UNTIL_PENALTY_S

        self.penalty_timer = 0.0
        self.penalty_side = None

    def __call__(self, mdp, state, action, next_state, absorbing):
        r = 0.0
        puck_pos, puck_vel = mdp.get_puck(next_state)

        ## Penalty checking (timer update)
        # Determine which side the puck is on on the first step
        if self.penalty_side is None:
            self.penalty_side = np.sign(puck_pos[0])

        if np.sign(puck_pos[0]) == self.penalty_side:
            # If the puck is on the same side as the penalty side, increment the penalty timer
            self.penalty_timer += mdp.env_info["dt"]
        else:
            # Otherwise, reset the penalty timer and change the penalty side
            self.penalty_side *= -1
            self.penalty_timer = 0.0
        ## ~ Penalty checking (timer update)

        if absorbing or mdp._data.time < mdp.env_info["dt"] * 2:
            ## Penalty checking
            # If the penalty timer is greater than X seconds and the puck is not in the middle, give reward accordingly
            if (
                self.penalty_timer > self._penalty_threshold
                and np.abs(puck_pos[0]) >= 0.15
            ):
                if self.penalty_side == -1:
                    r = self._reward_agent_faul
                elif self.penalty_side == 1:
                    r = self._reward_opponent_faul
                else:
                    raise ValueError(
                        f"Penalty side should be either -1 or 1, but got {self.penalty_side}"
                    )
            ## ~ Penalty checking

            ## Puck stuck in the middle
            if np.abs(puck_pos[0]) < 0.15 and np.abs(puck_vel[0]) < 0.025:
                r = self._reward_agent_cause_puck_stuck
            ## ~ Puck stuck in the middle

            ## Goal checking
            if (np.abs(puck_pos[1]) - mdp.env_info["table"]["goal_width"] / 2) <= 0:
                if puck_pos[0] > mdp.env_info["table"]["length"] / 2:
                    r = self._reward_agent_score_goal
                elif puck_pos[0] < -mdp.env_info["table"]["length"] / 2:
                    r = self._reward_agent_receive_goal
            ## ~ Goal checking

            # Reset the penalty timer and side (it is the end of episode)
            self.penalty_timer = 0.0
            self.penalty_side = None

        return r


class HitReward:
    def __init__(self):
        self.has_hit = False

    def __call__(self, mdp, state, action, next_state, absorbing):
        r = 0
        # Get puck's position and velocity (The position is in the world frame, i.e., center of the table)
        puck_pos, puck_vel = mdp.get_puck(next_state)

        # Define goal position
        goal = np.array([0.98, 0])
        # Compute the vector that shoot the puck directly to the goal
        vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])

        # width of table minus radius of puck
        effective_width = (
            mdp.env_info["table"]["width"] / 2 - mdp.env_info["puck"]["radius"]
        )

        # Calculate bounce point by assuming incoming angle = outgoing angle
        w = (
            abs(puck_pos[1]) * goal[0]
            + goal[1] * puck_pos[0]
            - effective_width * puck_pos[0]
            - effective_width * goal[0]
        ) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)
        side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])

        # Compute the vector that shoot puck with a bounce to the wall
        vec_puck_side = (side_point - puck_pos[:2]) / np.linalg.norm(
            side_point - puck_pos[:2]
        )

        if not self.has_hit:
            self.has_hit = _has_hit(mdp, state)

        if absorbing or mdp._data.time < mdp.env_info["dt"] * 2:
            # If the hit scores
            if (
                (puck_pos[0] - mdp.env_info["table"]["length"] / 2)
                > 0
                > (np.abs(puck_pos[1]) - mdp.env_info["table"]["goal_width"] / 2)
            ):
                r = 50
            self.has_hit = False
        else:
            # If the puck has not yet been hit, encourage the robot to get closer to the puck
            if not self.has_hit:
                ee_pos = mdp.get_ee()[0][:2]
                dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos)

                vec_ee_puck = (puck_pos[:2] - ee_pos) / dist_ee_puck

                cos_ang_side = np.clip(vec_puck_side @ vec_ee_puck, 0, 1)

                # Reward if vec_ee_puck and vec_puck_goal have the same direction
                cos_ang_goal = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
                cos_ang = np.max([cos_ang_goal, cos_ang_side])

                r = -dist_ee_puck / 2 + (cos_ang - 1) * 0.5
            else:
                r = min([1, 0.3 * np.maximum(puck_vel[0], 0.0)])

                # Encourage the puck to end in the middle
                if puck_pos[0] > 0.7 and puck_vel[0] > 0.1:
                    r += 0.5 - np.abs(puck_pos[1])

                # penalizes the joint velocity
                q = mdp.get_joints(next_state, 1)[0]
                r -= 0.01 * np.linalg.norm(q - mdp.init_state)
        return r


class DefendReward:
    def __init__(self):
        self.has_hit = False

    def __call__(self, mdp, state, action, next_state, absorbing):
        r = 0
        puck_pos, puck_vel = mdp.get_puck(next_state)

        if absorbing or mdp._data.time < mdp.env_info["dt"] * 2:
            self.has_hit = False

        if not self.has_hit:
            self.has_hit = _has_hit(mdp, state)

        # This checks weather the puck is in our goal, heavy penalty if it is.
        # If absorbing the puck is out of bounds of the table.
        if absorbing:
            # puck position is behind table going to the negative side
            if (
                puck_pos[0] + mdp.env_info["table"]["length"] / 2 < 0
                and (np.abs(puck_pos[1]) - mdp.env_info["table"]["goal_width"] / 2) < 0
            ):
                r = -100
            elif np.linalg.norm(puck_vel[:1]) < 0.1:
                # If the puck velocity is smaller than the threshold, the episode terminates with a high reward
                r = 150
        else:
            # If the puck bounced off the head walls, there is no reward.
            if puck_pos[0] <= -0.85 or puck_vel[0] > 0.3:
                r = -1
            # if the puck has been hit, or bounced off the wall
            elif puck_vel[0] > -0.2:
                # Reward if the puck slows down on the defending side
                r = 0.3 - abs(puck_vel[0])
            else:
                # If we did not yet hit the puck, reward is controlled by the distance between end effector and puck
                # on the x axis
                ee_pos = mdp.get_ee()[0][:2]
                ee_des = np.array([-0.6, puck_pos[1]])
                dist_ee_puck = ee_des - ee_pos
                r = -np.linalg.norm(dist_ee_puck)

        # penalizes the joint velocity
        q = mdp.get_joints(next_state)[0]
        r -= 0.005 * np.linalg.norm(q - mdp.init_state)
        return r


class PrepareReward:
    def __init__(self):
        self.has_hit = False

    def __call__(self, mdp, state, action, next_state, absorbing):
        puck_pos, puck_vel = mdp.get_puck(next_state)
        puck_pos = puck_pos[:2]
        puck_vel = puck_vel[:2]
        ee_pos = mdp.get_ee()[0][:2]

        if absorbing or mdp._data.time < mdp.env_info["dt"] * 2:
            self.has_hit = False

        if not self.has_hit:
            self.has_hit = _has_hit(mdp, state)

        if absorbing and abs(puck_pos[1]) < 0.3 and -0.8 < puck_pos[0] < -0.2:
            return 10
        elif absorbing:
            return -10
        else:
            if not self.has_hit:
                # encourage make contact
                dist_ee_puck = np.linalg.norm(puck_pos - ee_pos)
                vec_ee_puck = (puck_pos - ee_pos) / dist_ee_puck
                if puck_pos[0] > -0.65:
                    cos_ang = np.clip(
                        vec_ee_puck @ np.array([0, np.copysign(1, puck_pos[1])]), 0, 1
                    )
                else:
                    cos_ang_side = np.clip(
                        vec_ee_puck
                        @ np.array(
                            [
                                np.copysign(0.05, -0.5 - puck_pos[0]),
                                np.copysign(0.8, puck_pos[1]),
                            ]
                        ),
                        0,
                        1,
                    )
                    cos_ang_bottom = np.clip(vec_ee_puck @ np.array([-1, 0]), 0, 1)
                    cos_ang = max([cos_ang_side, cos_ang_bottom])
                r = -dist_ee_puck / 2 + (cos_ang - 1) * 0.5
            else:
                if -0.5 < puck_pos[0] < -0.2 and puck_pos[1] < 0.3:
                    r = np.clip(np.abs(-0.5 - puck_pos[0]) / 2, 0, 1) + (
                        0.3 - np.abs(puck_pos[1])
                    )
                else:
                    r = 0

                q = mdp.get_joints(next_state)[0]
                r -= 0.005 * np.linalg.norm(q - mdp.init_state)
        return r


def _has_hit(mdp, state):
    ee_pos, ee_vel = mdp.get_ee()
    puck_cur_pos, _ = mdp.get_puck(state)
    if (
        np.linalg.norm(ee_pos[:2] - puck_cur_pos[:2])
        < mdp.env_info["puck"]["radius"] + mdp.env_info["mallet"]["radius"] + 5e-3
    ):
        return True
    else:
        return False
