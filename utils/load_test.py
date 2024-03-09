def load_test_args(dataset_type,question_type):
    if dataset_type == 'counterfactual':
        datafolder = 'data/counterfactual/'
        if question_type == 'simple':
            dataset_filename = "fake_kg_with_simple_questions.json"
            example_filename = "sampled_fake_kg_with_simple_questions.json"
            question_key = 'selected_simple_question'
        elif question_type == 'explicit':
            dataset_filename = "fake_kg_with_explicit_questions.json"
            example_filename = "sampled_fake_kg_with_explicit_questions.json"
            question_key = 'explicit_question'
        elif question_type == 'implicit':
            dataset_filename = "fake_kg_with_implicit_questions.json"
            example_filename = "sampled_fake_kg_with_implicit_questions.json"
            question_key = 'implicit_question'
        elif question_type == 'supplement':
            dataset_filename = "supplementary_fake_kg_with_implicit_questions.json"
            example_filename = "sampled_fake_kg_with_implicit_questions.json"
            question_key = 'implicit_question'
    elif dataset_type == 'longtail':
        datafolder = 'data/longtail/'
        if question_type == 'simple':
            dataset_filename = "longtail_simple_questions.json"
            example_filename = "sampled_longtail_simple_questions.json"
            question_key = 'simple_question'
        elif question_type == 'explicit':
            dataset_filename = "longtail_explicit_questions.json"
            example_filename = "sampled_longtail_explicit_questions.json"
            question_key = 'explicit_question'
        elif question_type == 'implicit':
            dataset_filename = "longtail_implicit_questions.json"
            example_filename = "sampled_longtail_implicit_questions.json"
            question_key = 'implicit_question'
    
    return datafolder,dataset_filename,example_filename,question_key