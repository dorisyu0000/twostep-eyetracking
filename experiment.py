import os
import logging
import json
import re
from datetime import datetime
import psychopy
from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import numpy as np

from util import jsonify
from config import KEY_CONTINUE
from trial import GraphTrial, AbortKeyPressed
from graphics import Graphics
from bonus import Bonus
from eyetracking import EyeLink, MouseLink

import subprocess
from copy import deepcopy
from config import VERSION

from triggers import Triggers

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
                self.win.showMessage('The experiment ran into a problem! Please tell the experimenter.\nThen press C to continue.')
                self.win.flip()
                event.waitKeys(keyList=['c'])
                self.win.showMessage(None)
                logging.warning(f'Retrying {stage}')
                wrapper(self, *args, **kwargs)
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


class Experiment(object):
    def __init__(self, config_number, name=None, full_screen=False, score_limit=None, time_limit=30, test_mode=False, **kws):
        if config_number is None:
            config_number = get_next_config_number()
        self.config_number = config_number
        print('>>>', self.config_number)
        self.full_screen = full_screen
        self.score_limit = score_limit
        self.time_limit = time_limit

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
        self.bonus = Bonus(1, 0)
        self.total_score = 0
        # self.bonus = Bonus(self.parameters['points_per_cent'], 50)
        self.eyelink = None
        self.disable_gaze_contingency = False

        self._message = visual.TextBox2(self.win, '', pos=(-.83, 0), color='white', autoDraw=True, size=(0.65, None), letterHeight=.035, anchor='left')
        self._tip = visual.TextBox2(self.win, '', pos=(-.83, -0.2), color='white', autoDraw=True, size=(0.65, None), letterHeight=.025, anchor='left')

        # self._practice_trials = iter(self.trials['practice'])
        self.practice_i = -1
        self.trial_data = []
        self.practice_data = []

        self.parameters['triggers'] = self.triggers = Triggers(**({'port': 'dummy'} if test_mode else {}))

    def _reset_practice(self):
        self._practice_trials = iter(self.trials['practice'])

    def get_practice_trial(self, repeat=False,**kws):
        if not repeat:
            self.practice_i += 1
        prm = {
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
        size = (1350,750) if self.full_screen else (900,500)
        win = visual.Window(size, allowGUI=True, units='height', fullscr=self.full_screen)
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
        self._tip.setText(tip_text if tip_text else 'press space to continue' if space else '')
        self.win.flip()
        if space:
            event.waitKeys(keyList=[KEY_CONTINUE])

    @stage
    def setup_eyetracker(self, mouse=False):
        self.message("We're going to calibrate the eyetracker. Please tell the experimenter.",
            tip_text="Wait for the experimenter (space)", space=True)
        self.hide_message()
        if mouse:
            self.eyelink = MouseLink(self.win, self.id)
        else:
            self.eyelink = EyeLink(self.win, self.id)
        self.eyelink.setup_calibration()
        self.eyelink.calibrate()

    @stage
    def show_gaze_demo(self):
        self.message("Check it out! This is where the eyetracker thinks you're looking.",
                     tip_text='press space to continue')

        self.eyelink.start_recording()
        while KEY_CONTINUE not in event.getKeys():
            visual.Circle(self.win, radius=.01, pos=self.eyelink.gaze_position(), color='red').draw()
            self.win.flip()
        self.win.flip()

    @stage
    def intro_gaze(self):
        self.message("At the beginning of each round, a circle will appear. "
                     "Look straight at it and press space to start the round.",
                     tip_text="look at the circle and press space", space=False)

        self.eyelink.drift_check()
        self.message("Yup just like that. Make sure you hold your gaze steady on the circle before pressing space.", space=True)

    @stage
    def practice(self):
        self.message("Before we begin the main phase, we'll do a few practice rounds with all the images visible.", space=True)
        self.message("")
        for i in range(5):
            gt = self.get_practice_trial()
            gt.run()

    @stage
    def intro_main(self):
        self.message("Alright! We're ready to begin the main phase of the experiment.", space=True)
        self.message(f"You will have {self.time_limit} minutes to make as many points as you can.", space=True)
        self.message("Just like before, the clock only runs when the board is on the screen.", space=True)
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
        visual.TextBox2(self.win, msg, color='white', letterHeight=.035).draw()
        self.win.flip()
        if space:
            event.waitKeys(keyList=[KEY_CONTINUE])

    @stage
    def run_main(self, n=None):
        seconds_left = self.time_limit * 60

        last_summary_time = seconds_left

        summarize_every = self.parameters.get('summarize_every', 60 * 3)
        summarize_every = 10

        trials = self.trials['main']
        if n is not None:
            trials = trials[:n]

        for (i, trial) in enumerate(trials):
            logging.info(f"Trial {i+1} of {len(trials)}  {round(seconds_left / 60)} minutes left")
            try:

                if (last_summary_time - seconds_left) > summarize_every:
                    last_summary_time = seconds_left
                    msg = f"{self.bonus.report_bonus()}\nYou have about {round(seconds_left / 60)} minutes left.\nPress any key to continue."
                    logging.info('summary message: %s', msg)
                    self.center_message(msg, space=False)
                    event.waitKeys()


                prm = {**self.parameters, **trial}
                if self.disable_gaze_contingency:
                    prm['gaze_contingent'] = False
                    prm['start_mode'] = 'fixation'

                gt = GraphTrial(self.win, **prm, hide_states=True, eyelink=self.eyelink)
                gt.run()
                seconds_left -= (gt.done_time - gt.show_time)
                logging.info("seconds left: %s", seconds_left)

                psychopy.logging.flush()
                self.trial_data.append(gt.data)

                logging.info('gt.status is %s', gt.status)
                self.bonus.add_points(gt.score)
                self.total_score += int(gt.score)

                if gt.status == 'recalibrate':
                    self.eyelink.calibrate()

            except Exception as e:
                if isinstance(e, AbortKeyPressed):
                    logging.warning("Abort key pressed")
                    msg = 'Abort key pressed!'
                else:
                    logging.exception(f"Caught exception in run_main")
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
        self.message("You're done! Let's just save your data...", tip_text="give us a few seconds", space=False)
        psychopy.logging.flush()

        fp = f'{DATA_PATH}/{self.id}.json'
        with open(fp, 'w') as f:
            f.write(jsonify(self.all_data))
        logging.info('wrote %s', fp)

        if self.eyelink:
            self.eyelink.save_data()
        self.message("Data saved! Please let the experimenter that you've completed the study.", space=True,
                    tip_text='press space to exit')

    def emergency_save_data(self):
        logging.warning('emergency save data')
        if self.eyelink:
            self.eyelink.save_data()
        logging.warning('eyelink data saved?')
        fp = f'{DATA_PATH}/{self.id}.txt'
        with open(fp, 'w') as f:
            f.write(str(self.all_data))
        logging.info('wrote %s', fp)



