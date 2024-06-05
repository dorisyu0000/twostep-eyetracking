from psychopy import core, visual, gui, data, event
import numpy as np
import logging
import json
from eyetracking import height2pix
from util import jsonify
import random

wait = core.wait

from config import COLOR_PLAN, COLOR_ACT, COLOR_LOSS, COLOR_WIN, COLOR_NEUTRAL, COLOR_HIGHLIGHT, KEY_CONTINUE, KEY_SWITCH, KEY_SELECT, KEY_ABORT

from graphics import Graphics, FRAME_RATE

def reward_color(r):
    return COLOR_WIN if r > 0 else COLOR_NEUTRAL if r == 0 else COLOR_LOSS

def reward_string(r):
    return f'{int(r):+}' if r else ''

def distance(p1, p2):
    (x1, y1), (x2, y2) = (p1, p2)
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


TRIGGERS = {
    'show description': 0,
    'show graph': 1,
    'start acting': 2,
    'done': 3,
}


class AbortKeyPressed(Exception): pass


class GraphTrial(object):
    """Graph navigation interface"""
    def __init__(self, win, graph, rewards, start, layout, pos=(0, 0), start_mode=None, max_score=None,
                 images=None, reward_info=None, reward_multiplier=2,
                 delayed_feedback=True, feedback_duration=3, action_time=2,
                 initial_stage='planning', hide_states=False, hide_rewards_while_acting=True, hide_edges_while_acting=True,
                 eyelink=None, triggers=None, **kws):
        self.win = win
        self.graph = graph
        self.rewards = list(rewards)
        self.start = start
        self.layout = layout
        self.plan_time = plan_time
        self.act_time = act_time
        if start_mode is None:
            start_mode = 'drift_check' if eyelink else 'space'
        self.start_mode = start_mode
        self.highlight_edges = highlight_edges
        self.stop_on_x = stop_on_x
        self.hide_rewards_while_acting = hide_rewards_while_acting

        self.eyelink = eyelink
        self.gaze_contingent = gaze_contingent
        self.gaze_tolerance = gaze_tolerance
        self.fixation_lag = fixation_lag
        self.show_gaze = show_gaze
        self.last_gaze = None
        self.pos = pos
        self.start_mode = start_mode or ('drift_check' if eyelink else 'space')
        self.max_score = max_score

        self.images = images
        self.reward_info = reward_info
        self.reward_multiplier = reward_multiplier

        self.delayed_feedback = delayed_feedback
        self.feedback_duration = feedback_duration
        self.action_time = action_time

        self.current_time = None
        self.start_time = None
        self.end_time = None

        self.stage = initial_stage
        self.hide_states = hide_states
        self.hide_rewards_while_acting = hide_states or hide_rewards_while_acting
        self.hide_edges_while_acting = hide_edges_while_acting

        self.eyelink = eyelink
        self.triggers = triggers

        self.status = 'ok'
        self.start_time = self.done_time = None
        self.disable_click = False
        self.score = 0
        self.current_state = None
        self.fixated = None
        self.fix_verified = None
        self.path = []
        self.data = {
            "trial": {
                "kind": self.__class__.__name__,
                "graph": graph,
                "rewards": rewards,
                "reward_info": reward_info,
                "initial_stage": initial_stage,
                "hide_states": hide_states,
                "hide_rewards_while_acting": hide_rewards_while_acting,
                "hide_edges_while_acting": hide_edges_while_acting,

                "start": start,
            },
            "events": [],
            "flips": [],
            "mouse": [],
        }
        logging.info("begin trial " + jsonify(self.data["trial"]))
        self.gfx = Graphics(win)
        self.mouse = event.Mouse()
        self.done = False

    def wait_keys(self, keys, time_limit=float('inf')):
        keys = event.waitKeys(maxWait=time_limit, keyList=[*keys, KEY_ABORT])
        if keys and KEY_ABORT in keys:
            self.status = 'abort'
            raise AbortKeyPressed()
        else:
            return keys


    def log(self, event, info={}):
        time = core.getTime()
        logging.debug(f'{self.__class__.__name__}.log {time:3.3f} {event} ' + ', '.join(f'{k} = {v}' for k, v in info.items()))
        datum = {
            'time': time,
            'event': event,
            **info
        }
        self.data["events"].append(datum)
        if self.triggers and event in TRIGGERS:
            self.triggers.send(TRIGGERS[event])
        if self.eyelink:
            self.eyelink.message(jsonify(datum), log=False)

    def reward_descriptions(self):
        def fmt(x):
            val, desc, targets = x["val"], x["desc"], x["targets"]
            val *= self.reward_multiplier
            return f'{val:+d} for {desc}'

        return [fmt(x) for x in self.reward_info]

    def show(self):
        self.show_time = core.getTime()
        self.log('show graph')
        self.gfx.rect((-.65, .45), .1, .1, fillColor='white')
        if hasattr(self, 'nodes'):
            self.gfx.show()
            for lab in self.reward_text:
                lab.autoDraw = False
            return

        self.nodes = nodes = []
        self.node_images = []
        for i, (x, y) in enumerate(self.layout):
            pos = 0.7 * np.array([x, y])
            nodes.append(self.gfx.circle(pos, name=f'node{i}', r=.05))
            self.node_images.append(self.gfx.image(pos, self.images[i], size=.08, autoDraw=not self.hide_states))
        self.data["trial"]["node_positions"] = [height2pix(self.win, n.pos) for n in self.nodes]

        self.arrows = {}
        for i, js in enumerate(self.graph):
            for j in js:
                self.arrows[(i, j)] = self.gfx.arrow(nodes[i], nodes[j])

        self.mask = self.gfx.rect((.1,0), 1.1, 1, fillColor='gray', opacity=0)
        self.gfx.shift(*self.pos)

        self.reward_labels = []
        # if self.reward_info:
        #     print("YO")
        #     descs = self.reward_descriptions()
        #     xs = (.4, -.4)

        #     for desc, x, color in zip(descs, (.45, -.45), (COLOR_WIN, COLOR_LOSS)):
        #         self.reward_labels.append(self.gfx.text(desc.replace('for', '\n'), (x, .4), color=color, height=.05))



    def hide(self):
        self.gfx.clear()

    def shift(self, x, y):
        self.gfx.shift(x, y)
        self.pos = np.array(self.pos) + [x, y]


    def set_reward(self, s, r):
        self.rewards[s] = r
        self.reward_labels[s].text = reward_string(r)

    def get_click(self):
        if self.mouse.getPressed()[0]:
            pos = self.mouse.getPos()
            for (i, n) in enumerate(self.nodes):
                if n.contains(pos):
                    return i

    def set_state(self, s):
        self.log('visit', {'state': s})
        self.nodes[s].fillColor = COLOR_PLAN if self.stage == 'planning' else COLOR_ACT
        self.score += self.rewards[s]
        prev = self.current_state

        # self.set_node_label(s, reward_string(self.rewards[s]))


        self.current_state = s
        if len(self.graph[self.current_state]) == 0:
            self.done = True

        if prev is None:  # initial
            self.node_images[s].setAutoDraw(False)

        else:  # not initial
            self.path.append(s)
            self.nodes[prev].fillColor = 'white'
            if not self.delayed_feedback:
                txt = visual.TextStim(self.win, reward_string(self.rewards[s]),
                    pos=self.nodes[s].pos + np.array([.06, .06]),
                    bold=True, height=.04, color=reward_color(self.rewards[s]))
                txt.setAutoDraw(True)
                if self.node_images[s]:
                    self.node_images[s].setAutoDraw(True)
                self.win.flip()
                txt.setAutoDraw(False)
                core.wait(1)
                self.node_images[s].setAutoDraw(False)

    def show_feedback(self):
        for s in self.path:
            visual.TextStim(self.win, reward_string(self.rewards[s]),
                pos=self.nodes[s].pos + np.array([.06, .06]),
                bold=True, height=.04, color=reward_color(self.rewards[s])).draw()
            self.node_images[s].setAutoDraw(True)

        self.win.flip()
        core.wait(self.feedback_duration)
        self.win.flip()

    def fade_out(self):
        self.mask.setAutoDraw(False); self.mask.setAutoDraw(True)  # ensure on top
        for p in self.gfx.animate(.3):
            self.mask.setOpacity(p)
            self.win.flip()
        self.gfx.clear()
        self.win.flip()
        # wait(.3)

    def maybe_drop_edges(self):
        if not self.graph[self.current_state]:
            return
        if random.random() < self.force_rate:
            print("forced!")
            forced = random.choice(self.graph[self.current_state])
            for j in self.graph[self.current_state]:
                if j != forced:
                    arrow = self.arrows.pop((self.current_state, j))
                    self.gfx.remove(arrow)

            self.graph[self.current_state] = [forced]

    def set_reward_display(self, on=True):
        for l in self.reward_labels:
            if l:
                l.setAutoDraw(on)

    def highlight_current_edges(self):
        for (i, j), arrow in self.arrows.items():
            if i == self.current_state:
                arrow.setAutoDraw(False); arrow.setAutoDraw(True)  # on top
                arrow.setColor(COLOR_HIGHLIGHT)
                self.nodes[j].setLineColor(COLOR_HIGHLIGHT)
            else:
                arrow.setColor('black')
                self.nodes[j].setLineColor('black')

    def get_move(self):
        choices = []
        arrows = []
        for (i, j), arrow in self.arrows.items():
            if i == self.current_state:
                arrow.setAutoDraw(False); arrow.setAutoDraw(True)  # on top
                arrows.append(arrow)
                choices.append(j)
            else:
                if self.hide_edges_while_acting:
                    arrow.setAutoDraw(False)
                else:
                    arrow.setColor('black')

        idx = random.randint(0, len(choices) - 1)
        arrows[idx].setColor(COLOR_HIGHLIGHT)
        self.win.flip()
        self.log('get move', {"selected": choices[idx]})

        deadline = core.getTime() + self.action_time
        while True:
            time_left = deadline - core.getTime()
            pressed = self.wait_keys([KEY_SELECT, KEY_SWITCH], time_limit=time_left)
            if pressed is None:
                self.log('timeout', {"selected": choices[idx]})
                arrows[idx].setColor('red')
                visual.TextStim(self.win, 'too slow!', pos=(0, 0), color='red', height=.035).draw()
                self.win.flip()
                core.wait(2)
                return choices[idx]
            elif KEY_SELECT in pressed:
                self.log('select', {"selected": choices[idx]})
                if self.hide_edges_while_acting:
                    for arrow in arrows:
                        arrow.setAutoDraw(False)
                return choices[idx]
            elif KEY_SWITCH in pressed:
                arrows[idx].setColor('black')
                idx = (idx + 1) % len(choices)
                arrows[idx].setColor(COLOR_HIGHLIGHT)
                self.log('switch', {"selected": choices[idx]})
                self.win.flip()
            else:
                assert False

    def start_recording(self):
        self.log('start recording')
        self.eyelink.start_recording()
        # TODO: draw reference
        # left = int(scn_width/2.0) - 60
        # top = int(scn_height/2.0) - 60
        # right = int(scn_width/2.0) + 60
        # bottom = int(scn_height/2.0) + 60
        # draw_cmd = 'draw_filled_box %d %d %d %d 1' % (left, top, right, bottom)
        # el_tracker.sendCommand(draw_cmd)


    def run_planning(self):
        self.log('start planning')
        self.stage = 'planning'
        self.nodes[self.current_state].fillColor = COLOR_PLAN

        keys = self.wait_keys([KEY_SWITCH, KEY_ABORT])

        if KEY_SWITCH in keys:
            self.log('end planning')

        elif KEY_ABORT in keys:
            logging.warning('abort key pressed')
            self.log('abort planning')
            self.status = 'abort'

    def hide_rewards(self):
        for img in self.node_images:
            if img:
                img.setAutoDraw(False)
        for lab in self.reward_labels:
            lab.setAutoDraw(False)

    def hide_edges(self):
        for a in self.arrows.values():
            a.setAutoDraw(False)
        # for i in range(len(self.nodes)):
        #     self.set_node_label(i, '')

    def run_acting(self, one_step):
        self.nodes[self.current_state].fillColor = COLOR_ACT
        self.log('start acting')
        if self.hide_rewards_while_acting:
            self.hide_rewards()
        if self.hide_edges_while_acting:
            self.hide_edges()
        self.stage = 'acting'

        while not self.done:
            s = self.get_move()
            self.set_state(s)
            if one_step:
                return
            self.win.flip()

    def show_description(self):
        self.log('show description')
        descs = self.reward_descriptions()
        xs = (.25, -.25)
        y = 0
        for i, x in enumerate(xs):
            visual.TextStim(self.win, descs[i], pos=(x, y), color='white', height=.05).draw()
            if not self.hide_states:
                targets = self.reward_info[i]["targets"]
                xs = np.arange(len(targets)) * .1
                xs -= xs.mean()
                xs += x
                for x, t in zip(xs, targets):
                    self.gfx.image((x, y-.1), self.images[t], size=.08, autoDraw=False).draw()

        self.win.flip()
        self.wait_keys([KEY_CONTINUE])

    def fixation_cross(self):
        visual.TextStim(self.win, "+", pos=self.pos, color='black', height=.05).draw()
        self.win.flip()
        self.wait_keys([KEY_CONTINUE])

    def run(self, one_step=False, skip_planning=False):
            # if self.start_mode == 'drift_check':
            #     self.log('begin drift_check')
            #     self.status = self.eyelink.drift_check(self.pos)
            # elif self.start_mode == 'fixation':
            #     self.log('begin fixation')
            #     self.status = self.eyelink.fake_drift_check(self.pos)
            # elif self.start_mode == 'space':
            #     self.log('begin space')
            #     visual.TextStim(self.win, f'press {KEY_CONTINUE.upper()} to start', pos=self.pos, color='white', height=.035).draw()
            #     self.win.flip()
            #     self.wait_keys(['space', KEY_CONTINUE])

        self.log('initialize', {'status': self.status})

        if self.status in ('abort', 'recalibrate'):
            self.log('done', {"status": self.status})
            return self.status

        if self.eyelink:
            self.start_recording()

        self.log('cross 1')
        self.fixation_cross()
        self.start_time = core.getTime()
        if self.reward_info:
            self.show_description()

        # self.log('cross 2')
        # self.fixation_cross()
        # self.eyelink.fake_drift_check(self.pos)
        self.show()

        if self.current_state is None:
            self.set_state(self.start)

        self.log('start', {'flip_time': self.win.flip()})

        if not (one_step or skip_planning):
            self.run_planning()
        if self.status == 'abort':
            return 'abort'

        self.run_acting(one_step)
        if self.status == 'abort':
            return 'abort'
        if one_step:
            return

        if self.delayed_feedback:
            wait(.7)
            self.nodes[self.current_state].fillColor = 'white'
            self.show_feedback()
        else:
            wait(.2)

        self.fade_out()

        self.log('done')
        self.done_time = core.getTime()
        logging.info("end trial " + jsonify(self.data["events"]))
        if self.eyelink:
            self.eyelink.stop_recording()
        return self.status
    
    class CalibrationTrial(GraphTrial):
        """docstring for CalibrationTrial"""
        all_failures = np.zeros(11)  # across separate runs ASSUME graph doesn't change

        def __init__(self, *args, saccade_time=.7, n_success=2, n_fail=3, target_delay=.3 , **kwargs):
            kwargs['gaze_contingent'] = True
            kwargs['fixation_lag'] = .1
            kwargs['end_time'] = None

            self.saccade_time = saccade_time
            self.n_success = n_success
            self.n_fail = n_fail
            self.target_delay = target_delay

            self.target = None
            self.last_target = None
            self.arrow = None
            self.result = None

            super().__init__(*args, **kwargs)

        def node_label(self, i):
            return {
                # self.completed: ''
                # self.fixated: '',
                self.target: 'O',
            }.get(i, '')

        def do_timeout(self):
            self.log('timeout')
            logging.info('timeout')
            self.result = 'timeout'

        def draw_arrow(self):
            if self.arrow is not None:
                self.arrow.setAutoDraw(False)
            if self.last_target is not None:
                self.arrow = self.gfx.arrow(self.nodes[self.last_target], self.nodes[self.target])

        def new_target(self):
            initial = self.target is None
            self.last_target = self.target

            if initial:
                self.target = np.random.choice(len(self.successes))
            else:
                p = np.exp(
                    -5 * self.successes +
                    self.all_failures[:len(self.successes)]
                )
                p[self.target] = 0
                p /= (sum(p) or 1)  # prevent divide by 0
                self.target = np.random.choice(len(p), p=p)

            self.target_time = 'flip'  # updated to be next flip time
            self.draw_arrow()
            self.update_node_labels()
            self.log('new target', {"state": self.target})

        def tick(self):
            t = super().tick()
            if self.target_time == 'flip':
                self.target_time = t

        def run(self, timeout=15):
            assert self.eyelink
            # self.eyelink.drift_check(self.pos)
            self.start_recording()
            self.show()
            self.successes = np.zeros(len(self.nodes))
            self.failures = np.zeros(len(self.nodes))
            self.uncomplete = set(range(len(self.nodes)))
            self.new_target()
            self.start_time = self.tick()
            self.log('start', {'flip_time': self.start_time})

            self.win.mouseVisible = False

            self.target_time += 5  # extra time for first fixation
            while self.result is None:
                self.update_fixation()
                if 'x' in event.getKeys():  # cancel key
                    self.log('cancel')
                    self.result = 'cancelled'

                elif self.last_flip > self.target_time + self.saccade_time:  # timeout
                    self.log('timeout', {"state": self.target})
                    self.failures[self.target] += 1
                    self.all_failures[self.target] += 1

                    self.set_node_label(self.target, 'X')
                    lab = self.reward_labels[self.target]
                    for p in range(FRAME_RATE):
                        lab.setOpacity(1 - (p // 10) % 2)
                        self.tick()
                    wait(self.target_delay)
                    lab.setOpacity(1)

                    if sum(self.failures) == self.n_fail or self.failures[self.target] == 2:
                        self.result = 'failure'
                    else:
                        self.new_target()

                elif self.fixated == self.target:  # fixated within time
                    self.log('fixated target', {"state": self.target})
                    self.successes[self.target] += 1

                    lab = self.reward_labels[self.target]
                    for p in self.gfx.animate(6/60):
                        lab.setHeight(0.04 + p * 0.02)
                        self.tick()
                    for p in self.gfx.animate(12/60):
                        lab.setHeight(0.06 - p * 0.03)
                        lab.setOpacity(1-p)
                        self.tick()
                    wait(self.target_delay)
                    lab.setOpacity(1)
                    lab.setHeight(.03)

                    if self.successes[self.target] == self.n_success:
                        self.uncomplete.remove(self.target)
                    if self.uncomplete:
                        self.new_target()
                    else:
                        self.result = 'success'

                # if not self.done and self.end_time is not None and self.start_time + self.end_time < core.getTime():
                #     self.do_timeout()


                t = self.tick()

            self.log('done')
            self.eyelink.stop_recording()
            wait(.3)
            self.fade_out()
            self.win.mouseVisible = True

            return self.result
