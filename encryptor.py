import argparse


def encode(namespace):
    cipher = namespace.cipher
    key = namespace.key
    input_file = namespace.input_file
    output_file = namespace.output_file
    if (cipher != 'caesar' and cipher != 'vigenere') or key is None:
        raise SyntaxError('Incorrect cipher type or no key')
    if cipher == 'caesar':
        if not key.isdigit():
            raise SyntaxError('Incorrect key')
        encode_caesar(int(key), input_file, output_file)
    elif cipher == 'vigenere':
        if not isinstance(key, str):
            raise SyntaxError('Incorrect key')
        else:
            if not key.isalpha():
                raise SyntaxError('Incorrect key')
            encode_vigenere(key, input_file, output_file)


def encode_caesar(key=None, input_file=None, output_file=None):
    if input_file is None:
        inp = input()
    else:
        f = open(input_file, 'r')
        inp = f.read()
        f.close()
    encoded_inp = ''
    for letter in inp:
        if letter.isalpha():
            if letter.isupper():
                encoded_inp += chr((ord(letter) - 65 + key) % 26 + 65)
            else:
                encoded_inp += chr((ord(letter) - 97 + key) % 26 + 97)
        else:
            encoded_inp += letter
    if output_file is None:
        print(encoded_inp)
    else:
        f = open(output_file, 'w')
        f.write(encoded_inp)
        f.close()
    return


def encode_vigenere(key=None, input_file=None, output_file=None):
    if input_file is None:
        inp = input()
    else:
        f = open(input_file, 'r')
        inp = f.read()
        f.close()
    num = len(inp) // len(key) + 1
    new_key = (key * num)[:len(inp)]
    encoded_inp = ''
    for code_letter, letter in zip(new_key, inp):
        if code_letter.isupper():
            code_letter_num = (ord(code_letter) - 65)
        else:
            code_letter_num = (ord(code_letter) - 97)
        if letter.isalpha():
            if letter.isupper():
                encoded_inp += chr((ord(letter) - 65 + code_letter_num) % 26 + 65)
            else:
                encoded_inp += chr((ord(letter) - 97 + code_letter_num) % 26 + 97)
        else:
            encoded_inp += letter
    if output_file is None:
        print(encoded_inp)
    else:
        f = open(output_file, 'w')
        f.write(encoded_inp)
        f.close()
    return


def decode(namespace):
    cipher = namespace.cipher
    key = namespace.key
    input_file = namespace.input_file
    output_file = namespace.output_file
    if (cipher != 'caesar' and cipher != 'vigenere') or key is None:
        raise SyntaxError('Incorrect cipher type or no key')
    if cipher == 'caesar':
        if not key.isdigit():
            raise SyntaxError('Incorrect key')
        decode_caesar(int(key), input_file, output_file)
    elif cipher == 'vigenere':
        if not isinstance(key, str):
            raise SyntaxError('Incorrect key')
        else:
            if not key.isalpha():
                raise SyntaxError('Incorrect key')
            decode_vigenere(key, input_file, output_file)
    return


def decode_caesar(key=None, input_file=None, output_file=None):
    if input_file is None:
        inp = input()
    else:
        f = open(input_file, 'r')
        inp = f.read()
        f.close()
    encoded_inp = ''
    for letter in inp:
        if letter.isalpha():
            if letter.isupper():
                encoded_inp += chr((ord(letter) - 65 - key % 26 + 26) % 26 + 65)
            else:
                encoded_inp += chr((ord(letter) - 97 - key % 26 + 26) % 26 + 97)
        else:
            encoded_inp += letter
    if output_file is None:
        print(encoded_inp)
    else:
        f = open(output_file, 'w')
        f.write(encoded_inp)
        f.close()
    return


def decode_vigenere(key=None, input_file=None, output_file=None):
    if input_file is None:
        inp = input()
    else:
        f = open(input_file, 'r')
        inp = f.read()
        f.close()
    num = len(inp) // len(key) + 1
    new_key = (key * num)[:len(inp)]
    print(new_key)
    encoded_inp = ''
    for code_letter, letter in zip(new_key, inp):
        if code_letter.isupper():
            code_letter_num = (ord(code_letter) - 65)
        else:
            code_letter_num = (ord(code_letter) - 97)
        if letter.isalpha():
            if letter.isupper():
                encoded_inp += chr((ord(letter) - 65 - code_letter_num + 26) % 26 + 65)
            else:
                encoded_inp += chr((ord(letter) - 97 - code_letter_num + 26) % 26 + 97)
        else:
            encoded_inp += letter
    if output_file is None:
        print(encoded_inp)
    else:
        f = open(output_file, 'w')
        f.write(encoded_inp)
        f.close()
    return


def train(namespace):
    import json
    model_file = namespace.model_file
    input_file = namespace.input_file
    if model_file is None:
        raise SyntaxError('No file with model')
    if input_file is None:
        text = input()
    else:
        f = open(input_file, 'r')
        text = f.read()
        f.close()
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    model = {}
    for i in alphabet:
        model[i] = 0
    for letter in text:
        if letter.isalpha():
            model[letter] += 1
    f = open(model_file, 'w')
    json.dump(model, f)
    f.close()
    return


def normalize_dict(a):
    sum_values = sum(a.values())
    for i in a.keys():
        a[i] /= sum_values
    return a


def comparing_func(a, b, key):
    ans = 0
    for letter in a.keys():
        if letter.isupper():
            new_letter = chr((ord(letter) - 65 + 26 + key) % 26 + 65)
        else:
            new_letter = chr((ord(letter) - 97 + 26 + key) % 26 + 97)
        ans += abs(a[letter] - b[new_letter])
    return ans


def hack(namespace):
    input_file = namespace.input_file
    output_file = namespace.output_file
    model = namespace.model_file
    if model is None:
        raise SyntaxError('No file with model')
    import json
    if input_file is None:
        text = input()
    else:
        f = open(input_file, 'r')
        text = f.read()
        f.close()
    f = open(model, 'r')
    model_dict = json.load(f)
    f.close()
    text_dict = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in alphabet:
        text_dict[i] = 0
    for letter in text:
        if letter.isalpha():
            text_dict[letter] += 1
    low_score = 100000
    key = -1
    model_dict = normalize_dict(model_dict)
    text_dict = normalize_dict(text_dict)
    for i in range(26):
        score = comparing_func(model_dict, text_dict, i)
        if low_score > score:
            low_score = score
            key = i
    decode_caesar(key, input_file, output_file)
    return key


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='List of commands')

# encode command parser
encode_parser = subparsers.add_parser('encode', help='Encode input message')
encode_parser.add_argument('-c', '--cipher', required=True, type=str, help='Type of cipher: Caesar or Vigenere')
encode_parser.add_argument('-k', '--key', required=True, type=str, help='Cipher key')
encode_parser.add_argument('--input', type=str, dest='input_file', help='Path to the input file')
encode_parser.add_argument('--output', type=str, dest='output_file', help='Path to the output file')
encode_parser.set_defaults(func=encode)

# decode command parser
decode_parser = subparsers.add_parser('decode', help='Decode input message')
decode_parser.add_argument('-c', '--cipher', required=True, type=str, help='Type of cipher: Caesar or Vigenere')
decode_parser.add_argument('-k', '--key', required=True, type=str, help='Cipher key')
decode_parser.add_argument('--input', type=str, dest='input_file', help='Path to the input file')
decode_parser.add_argument('--output', type=str, dest='output_file', help='Path to the output file')
decode_parser.set_defaults(func=decode)

# train command parser
train_parser = subparsers.add_parser('train', help='Train language model on given text')
train_parser.add_argument('-m', '--model', required=True, type=str, dest='model_file', help='File where ' +
                                                                             'language model will be written')
train_parser.add_argument('--input', type=str, dest='input_file', help='Path to the input file')
train_parser.set_defaults(func=train)

# hack command parser
hack_parser = subparsers.add_parser('hack', help='Try to hack caesar code')
hack_parser.add_argument('-m', '--model', required=True, type=str, dest='model_file', help='File where ' +
                                                                             'language model will be taken from')
hack_parser.add_argument('--input', type=str, dest='input_file', help='Path to the input file')
hack_parser.add_argument('--output', type=str, dest='output_file', help='Path to the output file')
hack_parser.set_defaults(func=hack)


def test_accuracy():
    args = parser.parse_args('train --input shakespeare.txt --model model.json'.split())
    args.func(args)

    right = ''
    for i in range(26):
        args = parser.parse_args('encode --cipher caesar --key {} --input big.txt --output output.txt'.format(i).split())
        args.func(args)

        args = parser.parse_args('hack --input output.txt --model model.json --output trash.txt'.split())
        ans = args.func(args)

        if ans == i:
            right += ' ' + str(i)
    print(len(right.split()) / 26, right)


args = parser.parse_args()
if 'func' not in vars(args):
    raise SyntaxError('No functions')
args.func(args)

#test_accuracy()




