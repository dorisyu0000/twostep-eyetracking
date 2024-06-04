import os
import logging
import json
import re
from datetime import datetime
import psychopy
from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import numpy as np

from time import sleep
from util import jsonify
from config import KEY_CONTINUE, KEY_SWITCH, LABEL_CONTINUE, LABEL_SWITCH, LABEL_SELECT,COLOR_ACT
from trial import GraphTrial, AbortKeyPressed
from graphics import Graphics
from bonus import Bonus
from eyetracking import EyeLink, MouseLink

import subprocess
from copy import deepcopy
from config import VERSION

from triggers import Triggers
import hackfix

DATA_PATH = f'data/exp/{VERSION}'
CONFIG_PATH = f'config/{VERSION}'
LOG_PATH = 'log'
PSYCHO_LOG_PATH = 'psycho-log'
for p in (DATA_PATH, CONFIG_PATH, LOG_PATH, PSYCHO_LOG_PATH):
    os.makedirs(p, exist_ok=True)

def stage(f):
    def wrapper(self, *args, **kwargs):
        self.win.clearAutoDraw()
        logging.info('begin %s', f.__name__)
        try:
            f(self, *args, **kwargs)
        except:
            stage = f.__name__
            logging.exception(f"Caught exception in stage {stage}")
            if f.__name__ == "run_main":
                logging.warning('Continuing to save data...')
            else:
                self.win.clearAutoDraw()
                self.win.showMessage('The experiment ran into a problem! Press C to continue or Q to quit and save data')
                self.win.flip()
                keys = event.waitKeys(keyList=['c', 'q'])
                self.win.showMessage(None)
                if 'c' in keys:
                    logging.warning(f'Retrying {stage}')
                    wrapper(self, *args, **kwargs)
                else:
                    raise
        finally:
            self.win.clearAutoDraw()
            self.win.flip()

    return wrapper

def get_next_config_number():
    used = set()
    for fn in os.listdir(DATA_PATH):
        m = re.match(r'.*_P(\d+)\.', fn)
        if m:
            used.add(int(m.group(1)))

    possible = range(0, 1 + len(os.listdir(CONFIG_PATH)))
    try:
        n = next(i for i in possible if i not in used)
        return n
    except StopIteration:
        print("WARNING: USING RANDOM CONFIGURATION NUMBER")
        return np.random.choice(list(possible))


def text_box(win, msg, pos, autoDraw=True, wrapWidth=.8, height=.035, alignText='left', **kwargs):
    stim = visual.TextStim(win, msg, pos=pos, color='white', wrapWidth=wrapWidth, height=height, alignText=alignText, anchorHoriz='left', **kwargs)
    stim.autoDraw = autoDraw
    return stim
    import IPython, time; IPython.embed(); time.sleep(0.5)

class Experiment(object):
    def __init__(self, config_number, name=None, full_screen=False, score_limit=None, n_block=3, block_duration=10, n_practice=10, test_mode=False, **kws):
        if config_number is None:
            config_number = get_next_config_number()
        self.config_number = config_number
        print('>>>', self.config_number)
        self.full_screen = full_screen
        self.score_limit = score_limit
        self.n_block = n_block
        self.block_duration = block_duration
        self.n_practice = n_practice

        timestamp = datetime.now().strftime('%y-%m-%d-%H%M')
        self.id = f'{timestamp}_P{config_number}'
        if name:
            self.id += '-' + str(name)

        self.setup_logging()
        logging.info('git SHA: ' + subprocess.getoutput('git rev-parse HEAD'))

        config_file = f'{CONFIG_PATH}/{config_number}.json'
        logging.info('Configuration file: ' + config_file)
        with open(config_file) as f:
            conf = json.load(f)
            self.trials = conf['trials']
            self.parameters = conf['parameters']
        self.parameters.update(kws)
        logging.info('parameters %s', self.parameters)

        if 'gaze_tolerance' not in self.parameters:
            self.parameters['gaze_tolerance'] = 1.5

        self.win = self.setup_window()
        self.eyelink = MouseLink(self.win, self.id)  # use mouse by default
        self.win._heldDraw = []  # see hackfix
        self.bonus = Bonus(1, 0)
        self.total_score = 0
        # self.bonus = Bonus(self.parameters['points_per_cent'], 50)
        self.disable_gaze_contingency = False

        self._message = text_box(self.win, '', pos=(-0.4, 0.1), autoDraw=True, height=.035)
        self._tip = text_box(self.win, '', pos=(-0.4, -0.05), autoDraw=True, height=.025)

        # self._practice_trials = iter(self.trials['practice'])
        self.main_trials = iter(self.trials['main'])
        self.practice_i = -1
        self.trial_data = []
        self.practice_data = []
        self.parameters['triggers'] = self.triggers = Triggers(**({'port': 'dummy'} if test_mode else {}))

    def get_practice_trial(self, repeat=False,**kws):
        if not repeat:
            self.practice_i += 1
        prm = {
            'eyelink': self.eyelink,
            **self.parameters,
            # 'gaze_contingent': False,
            # 'time_limit': None,
            # 'pos': (.3, 0),
            # 'start_mode': 'immediate',
            # 'space_start': False,
            **self.trials['practice'][self.practice_i],
            **kws
        }
        gt = GraphTrial(self.win, **prm)
        self.practice_data.append(gt.data)
        return gt


    @property
    def n_trial(self):
        return len(self.trials['main'])

    def setup_logging(self):
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
        rootLogger = logging.getLogger()
        rootLogger.setLevel('DEBUG')

        fileHandler = logging.FileHandler(f"{LOG_PATH}/{self.id}.log")
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel('INFO')
        rootLogger.addHandler(consoleHandler)

        logging.info(f'starting up {self.id} at {core.getTime()}')

        psychopy.logging.LogFile(f"{PSYCHO_LOG_PATH}/{self.id}-psycho.log", level=logging.INFO, filemode='w')
        psychopy.logging.log(datetime.now().strftime('time is %Y-%m-%d %H:%M:%S,%f'), logging.INFO)


    def setup_window(self):
        size = (1350,750) if self.full_screen else (650,500)
        win = visual.Window(size, allowGUI=True, units='height', fullscr=self.full_screen)
        logging.info(f'Created window with size {win.size}')
        # framerate = win.getActualFrameRate(threshold=1, nMaxFrames=1000)
        # assert abs(framerate - 60) < 2
        win.flip()
        # win.callOnFlip(self.on_flip)
        return win

    def on_flip(self):
        if 'q' in event.getKeys():
            exit()
        # if 'f' in event.getKeys():

        self.win.callOnFlip(self.on_flip)

    def hide_message(self):
        self._message.autoDraw = False
        self._tip.autoDraw = False
        self.win.flip()

    def show_message(self):
        self._message.autoDraw = True
        self._tip.autoDraw = True

    def message(self, msg, space=False, tip_text=None):
        logging.debug('message: %s (%s)', msg, tip_text)
        self.show_message()
        self._message.setText(msg)
        self._tip.setText(tip_text if tip_text else f'press {LABEL_CONTINUE} to continue' if space else '')
        self.win.flip()
        if space:
            event.waitKeys(keyList=['space', KEY_CONTINUE])

    
    # @stage

    # def welcome(self):
    #     self.triggers.send(4)
    #     self.message(
    #         "Before we start, let's review the buttons. "
    #         f"{LABEL_CONTINUE} is the blue one. It should be under your right index finger. "
    #         f"You can press to confirm your choice", space=True
    #     )
    #     self.message(
    #         f"{LABEL_SWITCH} is the yellow one. It should be under your left index finger. "
    #         f"You can press to switch between the two options", 
    #         tip_text = f'press {LABEL_SWITCH} to continue')
    #     event.waitKeys(keyList=[KEY_SWITCH])

    @stage
    def intro(self):
        # self.message('Welcome!', space=True)
        gt = self.get_practice_trial(highlight_edges=False, hide_rewards_while_acting=False, initial_stage='acting')
        gt.show()

        gt.set_reward_display(False)
        self.message("In this experiment, you will play a game on the board shown to the right.", space=True)

        gt.set_state(gt.start)
        self.message("Your current location on the board is highlighted in blue.", space=True)

        gt.set_reward_display(True)
        self.message("The goal of the game is to collect these diamonds.", space=True)

        for (n, r) in zip(gt.nodes, gt.rewards):
            if r > 0:
                n.setLineColor('#1BD30C')
        self.message("Specifically, you want the ones that point to the right. These earn you points.", space=True)

        for (n, r) in zip(gt.nodes, gt.rewards):
            if r < 0:
                n.setLineColor('#E3000A')
            else:
                n.setLineColor('black')
        self.message("The diamonds that point left are bad. They take away points!", space=True)

        for (n, r) in zip(gt.nodes, gt.rewards):
            n.setLineColor('black')

        self.message("The further the diamond points to either side, the more points it is worth.", space=True)
        self.message("Hover over each diamond to see its point value",
                     tip_text='hover over every diamond to cotntinue', space=False)


        seen = set()
        n_reward = sum(l is not None for l in gt.reward_labels)
        while len(seen) < n_reward:
            pos = gt.mouse.getPos()
            for (i, n) in enumerate(gt.nodes):
                if gt.reward_labels[i]:
                    hovered = n.contains(pos)
                    if hovered:
                        seen.add(i)
                    gt.reward_labels[i].autoDraw = not hovered
                    gt.reward_text[i].autoDraw = hovered
            self.win.flip()
        sleep(0.5)



        
        self.message(
            "Before we start, let's review the buttons. "
            f"{LABEL_SELECT} is the blue one. It should be under your right index finger. "
            f"You can press to confirm your choice", space=False
        )
        self.message(
            f"{LABEL_SWITCH} is the yellow one. It should be under your left index finger. "
            f"You can press to switch between the two options", 
            tip_text = f'try to press {LABEL_SWITCH} and {LABEL_SELECT} to continue',space=False)
        
        gt.run(one_step=True)
        gt.start = gt.current_state

        self.message("The round ends when you get to a location with no outgoing connections.",
                     tip_text='click one of the highlighted locations', space=False)
        gt.run(skip_planning=True)

    @stage
    def practice_start(self):
        gt = self.get_practice_trial()
        gt.show()
        gt.set_state(gt.start)

        self.message("At the beginning of each round, your initial location will be red.", space=True)

        self.message("Before you can move, you have to click the red circle.", space=False,
                     tip_text='click the red circle to continue')
        gt.nodes[gt.start].setLineColor('#FFC910')
        gt.run_planning()
        gt.nodes[gt.start].setLineColor('black')

        gt.nodes[gt.start].fillColor = COLOR_ACT
        self.message("It will turn blue, indicating that you have entered the movement phase.", space=True)

        gt.hide_rewards()
        self.message("But be warned! The points will also disappear!", space=True)

        gt.update_node_labels()
        gt.nodes[gt.start].fillColor = COLOR_PLAN
        self.message("So, you should only enter the movement phase after deciding on a full path.", space=True)
        self.message("Give it a shot!", tip_text='click the red circle', space=False)

        gt.start_time = gt.tick()
        gt.run_planning()
        self.message("Now you can select which locations to visit.",
                     tip_text='complete the round to continue', space=False)
        gt.run(skip_planning=True)



    @stage
    def practice_change(self):
        gt = self.get_practice_trial()

        self.message("Both the connections and points change on every round of the game.",
                     tip_text='complete the round to continue', space=False)
        gt.run()

    @stage
    def practice_timelimit(self):
        gt = self.get_practice_trial(time_limit=3)
        gt.disable_click = True

        self.message("To make things more exciting, each round has a time limit.", space=True)
        gt.show()
        gt.timer.setLineColor('#FFC910')
        gt.timer.setLineWidth(5)
        gt.win.flip()

        self.message("The time left is indicated by a bar on the right.", space=True)
        gt.timer.setLineWidth(0)
        self.message("Let's see what happens when it runs out...", space=False,
            tip_text='wait for it')
        gt.run()
        self.message("If you run out of time, we'll make random decisions for you. Probably something to avoid.", space=True)

    @stage
    def practice(self, n):
        intervened = False
        for i in range(n):
            self.message("Let's try a few more practice rounds.",
                         space=False, tip_text=f'complete {n - i} practice rounds to continue')

            gt = self.get_practice_trial()
            for i in range(3):
                gt.run()
                if intervened or gt.score == gt.max_score:
                    break
                else:
                    self.message(
                        f"You got {int(gt.score)} points on that round, but you could have gotten {int(gt.max_score)}.\n"
                        f"Let's try again. Try to make as many points as possible!"
                    )
                gt = self.get_practice_trial(repeat=True)
            else:  # never succeeded
                logging.warning(f"failed practice trial {i}")
                if not intervened:
                    intervened = True
                    self.message("Please check in with the experimenter",
                        tip_text="Wait for the experimenter (space)", space=True)
                    self.get_practice_trial(repeat=True).run()


        self.message("Great job!", space=True)

    @stage
    def setup_eyetracker(self, mouse=False):
        self.message("Now we're going to calibrate the eyetracker. When you see a black "
                      "circle, look at it and hold your gaze steady", tip_text="wait for the experimenter")
        event.waitKeys(keyList=['space', 'c'])
        self.hide_message()
        if not mouse:
            self.eyelink = EyeLink(self.win, self.id)
        self.eyelink.setup_calibration()
        self.eyelink.calibrate()

    @stage
    def show_gaze_demo(self):
        self.message("Check it out! This is where the eyetracker thinks you're looking.",
                     tip_text=f'press {LABEL_CONTINUE} to continue')

        self.eyelink.start_recording()
        while KEY_CONTINUE not in event.getKeys():
            visual.Circle(self.win, radius=.01, pos=self.eyelink.gaze_position(), color='red').draw()
            self.win.flip()
        self.win.flip()

    # @stage
    # def intro_gaze(self):
    #     self.message("At the beginning of each round, a circle will appear. "
    #                  f"Look straight at it and press {LABEL_CONTINUE} to start the round.",
    #                  tip_text=f"look at the circle and press {LABEL_CONTINUE}", space=False)

    #     self.eyelink.drift_check()
    #     self.message("Yup just like that. Make sure you hold your gaze steady on the circle before pressing space.", space=True)

    @stage
    def practice(self):
        self.message("Before we begin the main phase, we'll do a few practice rounds with all the images visible.", space=True)
        self.hide_message()
        i = 0
        gt = self.get_practice_trial()
        while i < self.n_practice:
            try:
                logging.info('practice %s', i)
                gt.run()
                i += 1
                gt = self.get_practice_trial()
            except AbortKeyPressed:
                gt = self.get_practice_trial(repeat=True)
                self.win.clearAutoDraw()
                self.win.showMessage('Abort key pressed!\nPress C to continue, R to recalibrate, or Q to terminate the experiment and save data')
                self.win.flip()
                logging.warning('ABORT in practice, i=%s', i)
                keys = event.waitKeys(keyList=['c', 'r', 'q'])
                self.win.showMessage(None)
                self.win.flip()
                if 'c' in keys:
                    continue
                elif 'r' in keys:
                    self.eyelink.calibrate()
                else:
                    raise
                gt.gfx.clear()
                self.win.clearAutoDraw()
                self.win.flip()



    @stage
    def intro_main(self):
        self.message("Alright! We're ready to begin the main phase of the experiment.", space=True)
        self.message(f"There will be {self.n_block} blocks of {self.block_duration} minutes each.", space=True)
        self.message(f"Like before, the clock doesn't run in between rounds "
            "(when the cross is visible).", space=True)
        self.message(f"Remember, you will earn {self.bonus.describe_scheme()} you make the game.", space=True)
        self.message("Good luck!", space=True)
        # self.message("At the beginning of each round, look at the circle and press space.", space=True)

    @stage
    def run_one(self, i, **kws):
        trial = self.trials['main'][i]
        prm = {
            **self.parameters,
            **trial,
            **kws,
        }
        gt = GraphTrial(self.win, **prm, eyelink=self.eyelink)
        gt.run()
        self.bonus.add_points(gt.score)
        self.trial_data.append(gt.data)

    def center_message(self, msg, space=True):
        visual.TextStim(self.win, msg, color='white', wrapWidth=.8, alignText='center', height=.035).draw()
        self.win.flip()
        if space:
            event.waitKeys(keyList=[KEY_CONTINUE])

    def run_trial(self):
        trial = next(self.main_trials)
        prm = {**self.parameters, **trial}
        gt = GraphTrial(self.win, **prm, hide_states=True, eyelink=self.eyelink)
        gt.run()
        psychopy.logging.flush()
        self.trial_data.append(gt.data)

        logging.info('gt.status is %s', gt.status)
        self.bonus.add_points(gt.score)
        logging.info('current bonus: %s', self.bonus.dollars())
        self.total_score += int(gt.score)

        return core.getTime() - gt.start_time
        # if gt.status == 'recalibrate':
            # self.eyelink.calibrate()

    @stage
    def main(self, resume_block=None):
        start = 0 if resume_block is None else resume_block - 1

        for i in range(start, self.n_block):
            elapsed = 0

            while elapsed < 60 * self.block_duration:
                try:
                    elapsed += self.run_trial()
                    logging.info('elapsed is %s', elapsed)

                except Exception as e:
                    if isinstance(e, AbortKeyPressed):
                        logging.warning("Abort key pressed")
                        msg = 'Abort key pressed!'
                    else:
                        logging.exception(f"Caught exception in main")
                        msg = 'The experiment ran into a problem! Please tell the experimenter.'

                    self.win.clearAutoDraw()
                    self.win.showMessage(msg + '\n' + 'Press C to continue, R to recalibrate, or Q to terminate the experiment and save data')
                    self.win.flip()
                    keys = event.waitKeys(keyList=['c', 'r', 'q'])
                    self.win.showMessage(None)
                    if 'c' in keys:
                        continue
                    elif 'r' in keys:
                        self.eyelink.calibrate()
                    else:
                        raise

            # end while
            # block summary
            if i < self.n_block - 1:
                self.center_message(f"You've completed block {i + 1} of {self.n_block}.\n{self.bonus.report_bonus()}.\n\n"
                    "Take a short break. Then let the experimenter know when you're ready to continue.", space=False)
                event.waitKeys(keyList=['space', 'c'])
                self.eyelink.calibrate()


    @property
    def all_data(self):
        return {
            'config_number': self.config_number,
            'parameters': self.parameters,
            'trial_data': self.trial_data,
            'practice_data': self.practice_data,
            'window': self.win.size,
            'bonus': self.bonus.dollars()
        }

    @stage
    def save_data(self):
        self.message(f"You're done! {self.bonus.report_bonus('final')}",
                     tip_text="give us a few seconds to save the data", space=False)
        psychopy.logging.flush()

        fp = f'{DATA_PATH}/{self.id}.json'
        with open(fp, 'w') as f:
            f.write(jsonify(self.all_data))
        logging.info('wrote %s', fp)

        if self.eyelink:
            self.eyelink.save_data()

        self.message(f"You're done! {self.bonus.report_bonus('final')}",
                     tip_text="data saved! press Button 1 to exit", space=True)
        print("\n\nFINAL BONUS: ", self.bonus.dollars())

    def emergency_save_data(self):
        logging.warning('emergency save data')
        if self.eyelink:
            self.eyelink.save_data()
        logging.warning('eyelink data saved?')
        fp = f'{DATA_PATH}/{self.id}.txt'
        with open(fp, 'w') as f:
            f.write(str(self.all_data))
        logging.info('wrote %s', fp)



