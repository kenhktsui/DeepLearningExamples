import re
import sys

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        data = f.read()

    data = [i for i in data.split('\n') if i]
    unused = [i for i in data if re.search('\[unused[\d]+\]', i)]

    num_token = []
    for p in range(-3, 16):
        for i in range(0, 10):
            num_token.append(f'{i}@{p}')

    assert(len(num_token) < len(unused))

    replace_mapping = {org: new for org, new in zip(unused, num_token)}
    data = [replace_mapping[i] if i in replace_mapping else i for i in data]

    with open(output_path, 'w') as f:
        for i in data:
            f.write(i + '\n')
