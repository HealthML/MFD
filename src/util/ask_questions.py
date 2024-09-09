from .load_config import config

def print_question(question):
    print(f"\n# {question}?")

def get_input():
    return input('-> choose: ')

def multiple_choice_question(question, options):
    print_question(question)
    for i, option in enumerate(options):
        print(f'{i+1}) {option}')
    choice = get_input()
    assert choice in [str(i) for i in range(1, len(options)+1)]
    return options[int(choice)-1]

def int_question(question):
    print_question(question)
    choice = get_input()
    assert choice.isdigit()
    return int(choice)

def ask_questions(questions):
    answers = {}
    for question in questions:
        if question[1] == 'multiple choice':
            answers[question[0]] = multiple_choice_question(question[0], question[2])
        elif question[1] == 'int':
            answers[question[0]] = int_question(question[0])
        else:
            raise ValueError(f'question type {question[1]} not implemented')
    
    return answers

