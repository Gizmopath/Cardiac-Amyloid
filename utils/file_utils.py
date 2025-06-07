def load_test_cases_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    test_cases, is_test = [], False
    for line in lines:
        if line.startswith("Test Cases:"):
            is_test = True
            continue
        if is_test and line.strip():
            test_cases.append(line.strip())
    return test_cases
