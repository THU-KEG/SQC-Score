def load_prompt(model_name,shot_num=1):
    model_name = model_name.lower()
    prompt_folder = 'prompts/'
    if shot_num == 0:
        prompt_folder += 'zero_shot'
    elif shot_num == 1:
        prompt_folder += 'one_shot'
    elif shot_num == 2:
        prompt_folder += 'two_shot'
    elif shot_num == 3:
        prompt_folder += 'three_shot'
    elif shot_num == 4:
        prompt_folder += 'four_shot'
    
    if 'NeoXT-Chat'.lower() in model_name:
        return open(f'{prompt_folder}/chat_neox.txt').read()
    elif 'alpaca' in model_name:
        return open(f'{prompt_folder}/alpaca.txt').read()
    elif 'vicuna' in model_name:
        return open(f'{prompt_folder}/vicuna.txt').read()
    elif 'tulu' in model_name:
        return open(f'{prompt_folder}/tulu.txt').read()
    else:
        return open(f'{prompt_folder}/foundation_model.txt').read()