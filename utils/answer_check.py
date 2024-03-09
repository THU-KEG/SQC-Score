from Levenshtein import distance

def relative_distance(str1,str2):
    return distance(str1,str2)/(len(str1) + len(str2))

import re
def extract_answer(output_str,model_name):
    try:
        if 'flan' in model_name:
            # filter out special tokens as <(.*)> in output str
            output_str = re.sub(r"<(.*?)>","",output_str)
            if 'answer:' in output_str:
                return re.findall(r"answer: (.*)",output_str)[0]
            else:
                return output_str.strip()
        else:
            return re.findall(r"{{(.*?)}}",output_str)[0]
    except:
        return output_str.strip()
    
def get_answer_key(dataset_type,question_type):
    if dataset_type == 'counterfactual':
        if question_type == 'simple':
            answer_key = 'fake_answer_text'
        elif question_type == 'explicit':
            answer_key = 'fake_answer_text'
        elif question_type == 'implicit':
            answer_key = 'fake_answer_text'
    elif dataset_type == 'longtail':
        answer_key = 'answer_text'
    return answer_key

def check_answer(pred,ans):
    if type(ans) == list:
        # if any of the answers in the list
        # relative distance is less than 0.1
        # then return True
        for a in ans:
            if relative_distance(pred.lower(),a.lower()) < 0.1:
                return True
        return False
    else:
        return relative_distance(pred.lower(),ans.lower()) < 0.1

def check_answer_in_output(output_str,ans, stop_words):
    for stop_word in stop_words:
        output_str = output_str.split(stop_word)[0]
    if type(ans) == list:
        # if any of the answers in the list
        # relative distance is less than 0.1
        # then return True
        for a in ans:
            if a in output_str:
                return True
        return False
    else:
        return ans in output_str
    
def get_dataset_type_question_type_from_filename(filename):
    if 'fake_kg' in filename:
        dataset_type = 'counterfactual'
    elif 'longtail' in filename:
        dataset_type = 'longtail'
    if 'simple' in filename:
        question_type = 'simple'
    elif 'explicit' in filename:
        question_type = 'explicit'
    elif 'implicit' in filename:
        question_type = 'implicit'
    return dataset_type,question_type