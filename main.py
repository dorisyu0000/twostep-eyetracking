from experiment import Experiment
from fire import Fire
import logging

def main(config_number=None, name=None, test=False, fast=False, full=False, mouse=False, hotfix=False, **kws):
    if test and name is None:
        name = 'test'
    if fast:
        kws['score_limit'] = 10
    exp = Experiment(config_number, name, full_screen=(not test) or full, **kws)
    if test == "main":
        exp.run_main()
    elif test == "practice":
        exp.practice()
        exp.intro_main()
        exp.run_main()
    else:
        try:
            exp.setup_eyetracker(mouse)
            exp.show_gaze_demo()
            exp.intro_gaze()
            exp.practice()
            exp.intro_main()
            exp.run_main()
            exp.save_data()
        except:
            if test:
                exit(1)
            logging.exception('Uncaught exception in main')
            exp.win.clearAutoDraw()
            exp.win.showMessage("Drat! The experiment has encountered an error.\nPlease inform the experimenter.")
            exp.win.flip()
            try:
                exp.save_data()
                raise
            except:
                logging.exception('error on second save data attempt')
                exp.emergency_save_data()
                raise


if __name__ == '__main__':
    Fire(main)