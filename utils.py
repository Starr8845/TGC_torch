import torch as torch
import os
import logging
import logging


def string_format(x):
    out = ""
    if type(x)==dict:
        for key in x:
            if type(x[key])==list:
                out+= str(key)+': ' + ' '.join(['%.2e,'%(a) for a in x[key] ])
            else:
                out+=str(key)+ ': %.2e; '%(x[key])
    else:
        out = '%.2e'%(x)
    return out


class Logger(object):
    def __init__(self, info="", toy=False):
        self.info = info
        self.toy = toy
        self.folder_path, self.log_path = self.get_log_path()
        logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(message)s',
                datefmt='%m-%d %H:%M:%S',
                filename=os.path.join(self.log_path),
                filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        logging.getLogger('matplotlib.font_manager').disabled = True


    def get_log_path(self):
        path = "./logs"
        if self.toy:
            path = os.path.join(path, "toy_experiment")
        today, now = self.get_acc_time()
        if not os.path.exists(os.path.join(path, today)):
            os.mkdir(os.path.join(path, today))
        folder_path = os.path.join(path, today, f"{now}_{self.info}")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        log_path = os.path.join(folder_path, f"{now}_{self.info}.log")
        return folder_path, log_path

    def get_folder_path(self):
        return self.folder_path

    def get_acc_time(self):
        from datetime import timezone, timedelta, datetime

        beijing = timezone(timedelta(hours=8))
        utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        time_beijing = utc_time.astimezone(beijing)
        today = str(time_beijing.date())
        now = str(time_beijing.time())[:8]
        return today, now


if __name__=="__main__":
    a =  {'mse': 0.00039386216163351964, 'mrrt': 0.03724326581354963, 'btl': 0.9518861970718717, 'btl5': 1.1220883044356023, 'btl10': 1.1856731118605228, 'IC': 0.026117565814134207, 'ICIR': 0.28892131666548915, 'RIC': 0.028706416821933298, 'RICIR': 0.306045657114909,'long':[1.0,2,3,4]}

    # a = 0.00039386216163351964
    print(string_format(a))




