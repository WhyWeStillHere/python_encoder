import argparse
import sys

ENGLISH_ALPHABET_LEN = 26
ENGLISH_FIRST_LETTER_CODE = 97
ENGLISH_FIRST_CAPITAL_LETTER_CODE = 65
FIRST_VALID_LETTER = 32


def raw_text_to_text(raw, text):
    new_text = []
    j = 0
    for letter in text:
        if letter.isalpha():
            if letter.isupper():
                new_text += chr(ord(raw[j]) - FIRST_VALID_LETTER)
            else:
                new_text += raw[j]
            j += 1
        else:
            new_text += letter
    return ''.join(new_text)


def get_input(input_file):
    if input_file is None:
        inp = sys.stdin.read()
        print('\r')
    else:
        with open(input_file, 'r') as f:
            inp = f.read()
    return inp


def write_output(output, output_file):
    if output_file is None:
        print(output)
    else:
        with open(output_file, 'w') as f:
            f.write(output)


def encode(namespace):
    cipher = namespace.cipher
    key = namespace.key
    input_file = namespace.input_file
    output_file = namespace.output_file
    if key is None:
        raise SyntaxError('No key')
    if cipher == 'caesar':
        if not key.isdigit():
            raise SyntaxError('Incorrect key')
        encode_caesar(int(key), input_file, output_file)
    elif cipher == 'vigenere':
        if not isinstance(key, str):
            raise SyntaxError('Incorrect key')
        if not key.isalpha():
            raise SyntaxError('Incorrect key')
        encode_vigenere(key, input_file, output_file)
    elif cipher == 'vernam':
        if not isinstance(key, str):
            raise SyntaxError('Incorrect key')
        if not key.isalpha():
            raise SyntaxError('Incorrect key')
        encode_vernam(key, input_file, output_file)


def encode_caesar(key=None, input_file=None, output_file=None):
    inp = get_input(input_file)
    encoded_inp = []
    for letter in inp:
        if letter.isalpha():
            if letter.isupper():
                encoded_inp += chr((ord(letter) - ENGLISH_FIRST_CAPITAL_LETTER_CODE + key)
                                   % ENGLISH_FIRST_CAPITAL_LETTER_CODE + ENGLISH_FIRST_CAPITAL_LETTER_CODE)
            else:
                encoded_inp += chr((ord(letter) - ENGLISH_FIRST_LETTER_CODE + key)
                                   % ENGLISH_ALPHABET_LEN + ENGLISH_FIRST_LETTER_CODE)
        else:
            encoded_inp += letter
    encoded_inp = ''.join(encoded_inp)
    write_output(encoded_inp, output_file)
    return encoded_inp


def encode_vigenere(key=None, input_file=None, output_file=None):
    inp = get_input(input_file)
    text = [i for i in inp.lower() if i.isalpha()]
    num_of_blocks = len(inp) // len(key) + 1
    new_key = (key * num_of_blocks)[:len(text)]
    encoded_inp = []
    for code_letter, letter in zip(new_key, text):
        code_letter_num = (ord(code_letter) - ENGLISH_FIRST_LETTER_CODE)
        encoded_inp += chr((ord(letter) - ENGLISH_FIRST_LETTER_CODE + code_letter_num) % ENGLISH_ALPHABET_LEN
                           + ENGLISH_FIRST_LETTER_CODE)
    encoded_inp = raw_text_to_text(''.join(encoded_inp), inp)
    write_output(encoded_inp, output_file)
    return encoded_inp


def encode_vernam(key=None, input_file=None, output_file=None):
    # 20 because of problem with '/r' symbol
    first_letter_code = 20
    inp = get_input(input_file)
    encoded_inp = ''.join([chr(((ord(inp_letter) - first_letter_code) ^ (ord(key_letter) - first_letter_code))
                               + first_letter_code) for inp_letter, key_letter in zip(inp, key)])
    write_output(encoded_inp, output_file)
    return encoded_inp


def decode(namespace):
    cipher = namespace.cipher
    key = namespace.key
    input_file = namespace.input_file
    output_file = namespace.output_file
    if key is None:
        raise SyntaxError('No key')
    if cipher == 'caesar':
        if not key.isdigit():
            raise SyntaxError('Incorrect key')
        decode_caesar(int(key), input_file, output_file)
    elif cipher == 'vigenere':
        if not isinstance(key, str):
            raise SyntaxError('Incorrect key')
        if not key.isalpha():
            raise SyntaxError('Incorrect key')
        decode_vigenere(key, input_file, output_file)
    elif cipher == 'vernam':
        if not isinstance(key, str):
            raise SyntaxError('Incorrect key')
        if not key.isalpha():
            raise SyntaxError('Incorrect key')
        decode_vernam(key, input_file, output_file)


def decode_caesar(key=None, input_file=None, output_file=None):
    inp = get_input(input_file)
    encoded_inp = []
    for letter in inp:
        if letter.isalpha():
            if letter.isupper():
                encoded_inp += chr((ord(letter) - ENGLISH_FIRST_CAPITAL_LETTER_CODE - key % ENGLISH_ALPHABET_LEN
                                    + ENGLISH_ALPHABET_LEN) % ENGLISH_ALPHABET_LEN + ENGLISH_FIRST_CAPITAL_LETTER_CODE)
            else:
                encoded_inp += chr((ord(letter) - ENGLISH_FIRST_LETTER_CODE - key % ENGLISH_ALPHABET_LEN
                                    + ENGLISH_ALPHABET_LEN) % ENGLISH_ALPHABET_LEN + ENGLISH_FIRST_LETTER_CODE)
        else:
            encoded_inp += letter
    encoded_inp = ''.join(encoded_inp)
    write_output(encoded_inp, output_file)
    return encoded_inp


def decode_vigenere(key=None, input_file=None, output_file=None, text=None):
    if text is None:
        inp = get_input(input_file)
    else:
        inp = text
    text = [i for i in inp.lower() if i.isalpha()]
    num = len(inp) // len(key) + 1
    new_key = (key * num)[:len(text)]
    encoded_inp = []
    for code_letter, letter in zip(new_key, text):
        code_letter_num = (ord(code_letter) - ENGLISH_FIRST_LETTER_CODE)
        encoded_inp += chr((ord(letter) - ENGLISH_FIRST_LETTER_CODE - code_letter_num + ENGLISH_ALPHABET_LEN)
                           % ENGLISH_ALPHABET_LEN + ENGLISH_FIRST_LETTER_CODE)
    encoded_inp = raw_text_to_text(''.join(encoded_inp), inp)
    # text not None when we call decoder from hack function, so we don't need output
    if text is None:
        write_output(encoded_inp, output_file)
    return encoded_inp


def decode_vernam(key=None, input_file=None, output_file=None):
    # 20 because of problem with '/r' symbol
    first_letter_code = 20
    inp = get_input(input_file)
    encoded_inp = ''.join([chr(((ord(inp_letter) - first_letter_code) ^ (ord(key_letter) - first_letter_code))
                               + first_letter_code) for inp_letter, key_letter in zip(inp, key)])
    write_output(encoded_inp, output_file)
    return encoded_inp


def train(namespace):
    import json
    model_file = namespace.model_file
    input_file = namespace.input_file
    if model_file is None:
        raise SyntaxError('No file with model')
    text = get_input(input_file)
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    model = {}
    for i in alphabet:
        model[i] = 0
    for letter in text:
        if letter.isalpha():
            model[letter] += 1
    with open(model_file, 'w') as f:
        json.dump(model, f)


def normalize_dict(input_dict):
    sum_values = sum(input_dict.values())
    for i in input_dict.keys():
        input_dict[i] /= sum_values
    return input_dict


def comparing_func(first_dict, second_dict, key):
    ans = 0
    for letter in first_dict.keys():
        if letter.isupper():
            new_letter = chr((ord(letter) - ENGLISH_FIRST_CAPITAL_LETTER_CODE + ENGLISH_ALPHABET_LEN + key)
                             % ENGLISH_ALPHABET_LEN + ENGLISH_FIRST_CAPITAL_LETTER_CODE)
        else:
            new_letter = chr((ord(letter) - ENGLISH_FIRST_LETTER_CODE + ENGLISH_ALPHABET_LEN + key)
                             % ENGLISH_ALPHABET_LEN + ENGLISH_FIRST_LETTER_CODE)
        ans += abs(first_dict[letter] - second_dict[new_letter])
    return ans


def hack(namespace):
    input_file = namespace.input_file
    output_file = namespace.output_file
    model = namespace.model_file
    cipher = namespace.cipher
    if cipher == 'caesar':
        return hack_caesar(model, input_file, output_file)
    elif cipher == 'vigenere':
        return hack_vigenere(model, input_file, output_file)


def hack_caesar(model=None, input_file=None, output_file=None, text=None):
    if model is None:
        raise SyntaxError('No file with model')
    import json
    if text is None:
        text = get_input(input_file)
    with open(model, 'r') as f:
        model_dict = json.load(f)
    text_dict = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in alphabet:
        text_dict[i] = 0
    for letter in text:
        if letter.isalpha():
            text_dict[letter] += 1
    low_score = float('inf')
    key = -1
    model_dict = normalize_dict(model_dict)
    text_dict = normalize_dict(text_dict)
    for i in range(ENGLISH_ALPHABET_LEN):
        score = comparing_func(model_dict, text_dict, i)
        if low_score > score:
            low_score = score
            key = i
    if input_file is not None:
        decode_caesar(key, input_file, output_file)
    return key


# find key length with vigenere cipher
def find_key_len(text, match_index, alphabet_len):
    ans = -1
    eps = 1e-2
    for s in range(2, len(text) + 1):
        new_text = text[::s]
        if len(new_text) <= 1:
            break
        counter = [0] * alphabet_len
        for i in new_text:
            counter[ord(i) - ENGLISH_FIRST_LETTER_CODE] += 1

        counter = list(map(lambda x: (x * (x - 1)) / (len(new_text) * (len(new_text) - 1)), counter))
        if match_index - sum(counter) < eps:
            ans = s
            break
    return ans


def calc_mutual_match(first, second, alphabet_len, match_index):
    ans = -1
    eps = 1e-2
    for s in range(alphabet_len):
        index = 0
        for i in range(0, alphabet_len):
            index += first[i] * second[(i + s) % alphabet_len]
        index /= (sum(first) * sum(second))
        if match_index - index < eps:
            ans = s
            break
    return ans


# count different letters in string
def make_counter_dict(inp, alphabet_len):
    ans = [0] * alphabet_len
    for i in inp:
        ans[ord(i) - ENGLISH_FIRST_LETTER_CODE] += 1
    return ans


def hack_vigenere(model=None, input_file=None, output_file=None):
    text = get_input(input_file)
    alphabet_len = ENGLISH_ALPHABET_LEN
    match_index = 0.0644
    # create string only with letters
    new_text = [i for i in text.lower() if i.isalpha()]
    # find key length
    key_len = find_key_len(new_text, match_index, alphabet_len)
    # decompose text for key length
    new_table = [new_text[i::key_len] for i in range(0, key_len)]
    shift = [0] * (len(new_table) - 1)
    count = make_counter_dict(new_table[0], alphabet_len)
    # calculate relative letter arrangement
    for i in range(1, len(new_table)):
        shift[i - 1] = calc_mutual_match(count, make_counter_dict(new_table[i], alphabet_len),
                                         alphabet_len, match_index)
    # assume that first letter is 'a'
    key = ['a']
    for current_shift in shift:
        key += chr(current_shift + ENGLISH_FIRST_LETTER_CODE)
    key = ''.join(key)
    # decode text with new key
    decoded_text = decode_vigenere(key, text=text)
    # with hack_caesar find first letter for correct key
    step = hack_caesar(model, text=decoded_text)
    # calculate correct key
    key = [str(chr(step + ENGLISH_FIRST_LETTER_CODE))]
    for current_shift in shift:
        key += chr((step + current_shift) % ENGLISH_ALPHABET_LEN + ENGLISH_FIRST_LETTER_CODE)
    key = ''.join(key)
    # decode text with new key
    decode_vigenere(key, input_file, output_file)
    return key


def parse_args(command=None):
    # setting available cipher
    encode_ciphers = ['caesar', 'vigenere', 'vernam']
    decode_ciphers = ['caesar', 'vigenere', 'vernam']
    hack_ciphers = ['caesar', 'vigenere']

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='List of commands')

    # encode command parser
    encode_parser = subparsers.add_parser('encode', help='Encode input message')
    encode_parser.add_argument('-c', '--cipher', required=True, type=str, choices=encode_ciphers,
                               help='Type of cipher: Caesar or Vigenere or Vernam')
    encode_parser.add_argument('-k', '--key', required=True, type=str, help='Cipher key')
    encode_parser.add_argument('--input-file', type=str, dest='input_file', help='Path to the input file')
    encode_parser.add_argument('--output-file', type=str, dest='output_file', help='Path to the output file')
    encode_parser.set_defaults(func=encode)

    # decode command parser
    decode_parser = subparsers.add_parser('decode', help='Decode input message')
    decode_parser.add_argument('-c', '--cipher', required=True, type=str, choices=decode_ciphers,
                               help='Type of cipher: Caesar or Vigenere or Vernam')
    decode_parser.add_argument('-k', '--key', required=True, type=str, help='Cipher key')
    decode_parser.add_argument('--input-file', type=str, dest='input_file', help='Path to the input file')
    decode_parser.add_argument('--output-file', type=str, dest='output_file', help='Path to the output file')
    decode_parser.set_defaults(func=decode)

    # train command parser
    train_parser = subparsers.add_parser('train', help='Train language model on given text(Only for Caesar cipher)')
    train_parser.add_argument('-m', '--model-file', required=True, type=str,
                              dest='model_file', help='File where language model will be written')
    train_parser.add_argument('--text-file', required=True, type=str, dest='input_file',
                              help='Path to the input file')
    train_parser.set_defaults(func=train)

    # hack command parser
    hack_parser = subparsers.add_parser('hack', help='Try to hack Caesar or Vinegere code')
    hack_parser.add_argument('-c', '--cipher', default='caesar', type=str, choices=hack_ciphers,
                             help='Type of cipher: Caesar or Vigenere. Default = Caesar')
    hack_parser.add_argument('-m', '--model-file', required=True, type=str,
                             dest='model_file', help='File where language model will be taken from')
    hack_parser.add_argument('--input-file', type=str, dest='input_file', help='Path to the input file')
    hack_parser.add_argument('--output-file', type=str, dest='output_file', help='Path to the output file')
    hack_parser.set_defaults(func=hack)

    if command is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(command)
    if 'func' not in vars(args):
        raise SyntaxError('No functions')
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
