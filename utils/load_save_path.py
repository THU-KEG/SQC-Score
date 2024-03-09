import os
import datetime
def load_save_path(model_name,dataset_filename,shot_num,prompt_name,decode_type):
    model_name = model_name.lower()
    shot_str = ''
    if shot_num == 0:
        shot_str = 'zero_shot'
    elif shot_num == 1:
        shot_str = 'one_shot'
    elif shot_num == 2:
        shot_str = 'two_shot'
    elif shot_num == 3:
        shot_str = 'three_shot'
    elif shot_num == 4:
        shot_str = 'four_shot'

    path = f"data/results/{model_name}"
    if not os.path.exists(path):
        os.mkdir(path)
        
    if prompt_name != model_name:
        path += f"/{prompt_name}"
        if not os.path.exists(path):
            os.mkdir(path)
    
    if decode_type != 'greedy':
        path += f"/{decode_type}"
        if not os.path.exists(path):
            os.mkdir(path)
    
    path += f"/{shot_str}"
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    path += f"/{dataset_filename}"
    # check file confilict
    if os.path.exists(path):
        # ret with datetime postfix
        return path+f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        # ret without postfix
        return path
