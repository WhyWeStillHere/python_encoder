import argparse


def raw_text_to_text(raw, text):
    new_text = ''
    j = 0
    for letter in text:
        if letter.isalpha():
            if letter.isupper():
                new_text += chr(ord(raw[j]) - 32)
            else:
                new_text += raw[j]
            j += 1
        else:
            new_text += letter
    return new_text


def get_input(input_file):
    if input_file is None:
        inp = input()
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
    return


def encode(namespace):
    cipher = namespace.cipher
    key = namespace.key
    input_file = namespace.input_file
    output_file = namespace.output_file
    ciphers = ['caesar', 'vigenere', 'vernam']
    if cipher not in ciphers or key is None:
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
    elif cipher == 'vernam':
        if not isinstance(key, str):
            raise SyntaxError('Incorrect key')
        else:
            if not key.isalpha():
                raise SyntaxError('Incorrect key')
            encode_vernam(key, input_file, output_file)
    return


def encode_caesar(key=None, input_file=None, output_file=None):
    inp = get_input(input_file)
    encoded_inp = ''
    for letter in inp:
        if letter.isalpha():
            if letter.isupper():
                encoded_inp += chr((ord(letter) - 65 + key) % 26 + 65)
            else:
                encoded_inp += chr((ord(letter) - 97 + key) % 26 + 97)
        else:
            encoded_inp += letter
    write_output(encoded_inp, output_file)
    return encoded_inp


def encode_vigenere(key=None, input_file=None, output_file=None):
    inp = get_input(input_file)
    text = [i for i in inp.lower() if i.isalpha()]
    num = len(inp) // len(key) + 1
    new_key = (key * num)[:len(text)]
    encoded_inp = ''
    for code_letter, letter in zip(new_key, text):
        code_letter_num = (ord(code_letter) - 97)
        encoded_inp += chr((ord(letter) - 97 + code_letter_num) % 26 + 97)
    encoded_inp = raw_text_to_text(encoded_inp, inp)
    write_output(encoded_inp, output_file)
    return encoded_inp


def encode_vernam(key=None, input_file=None, output_file=None):
    inp = get_input(input_file)
    # + 20 because of problem with '/r' symbol
    encoded_inp = ''.join([chr(((ord(a) - 20) ^ (ord(b) - 20)) + 20) for a, b in zip(inp, key)])
    write_output(encoded_inp, output_file)
    return encoded_inp


def decode(namespace):
    cipher = namespace.cipher
    key = namespace.key
    input_file = namespace.input_file
    output_file = namespace.output_file
    ciphers = ['caesar', 'vigenere', 'vernam']
    if cipher not in ciphers or key is None:
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
    elif cipher == 'vernam':
        if not isinstance(key, str):
            raise SyntaxError('Incorrect key')
        else:
            if not key.isalpha():
                raise SyntaxError('Incorrect key')
            decode_vernam(key, input_file, output_file)
    return


def decode_caesar(key=None, input_file=None, output_file=None):
    inp = get_input(input_file)
    encoded_inp = ''
    for letter in inp:
        if letter.isalpha():
            if letter.isupper():
                encoded_inp += chr((ord(letter) - 65 - key % 26 + 26) % 26 + 65)
            else:
                encoded_inp += chr((ord(letter) - 97 - key % 26 + 26) % 26 + 97)
        else:
            encoded_inp += letter
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
    encoded_inp = ''
    for code_letter, letter in zip(new_key, text):
        code_letter_num = (ord(code_letter) - 97)
        encoded_inp += chr((ord(letter) - 97 - code_letter_num + 26) % 26 + 97)
    encoded_inp = raw_text_to_text(encoded_inp, inp)
    # text not None when we call decoder from hack function, so we don't need output
    if text is None:
        write_output(encoded_inp, output_file)
    return encoded_inp


def decode_vernam(key=None, input_file=None, output_file=None):
    inp = get_input(input_file)
    # + 20 because of problem with '/r' symbol
    encoded_inp = ''.join([chr(((ord(a) - 20) ^ (ord(b) - 20)) + 20) for a, b in zip(inp, key)])
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
    cipher = namespace.cipher
    ciphers = ['caesar', 'vigenere']
    if cipher not in ciphers:
        raise SyntaxError('Incorrect cipher type')
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
    low_score = 100000
    key = -1
    model_dict = normalize_dict(model_dict)
    text_dict = normalize_dict(text_dict)
    for i in range(26):
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
        counter = [0 for i in range(0, alphabet_len)]
        for i in new_text:
            counter[ord(i) - 97] += 1

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
    ans = [0 for i in range(alphabet_len)]
    for i in inp:
        ans[ord(i) - 97] += 1
    return ans


def hack_vigenere(model=None, input_file=None, output_file=None):
    text = get_input(input_file)
    alphabet_len = 26
    match_index = 0.0644
    # create string only with letters
    new_text = [i for i in text.lower() if i.isalpha()]
    # find key length
    key_len = find_key_len(new_text, match_index, alphabet_len)
    # decompose text for key length
    new_table = [new_text[i::key_len] for i in range(0, key_len)]
    shift = [0 for i in range(len(new_table) - 1)]
    count = make_counter_dict(new_table[0], alphabet_len)
    # calculate relative letter arrangement
    for i in range(1, len(new_table)):
        shift[i - 1] = calc_mutual_match(count, make_counter_dict(new_table[i], alphabet_len),
                                         alphabet_len, match_index)
    # assume that first letter is 'a'
    key = 'a'
    for j in range(len(shift)):
        key += chr(shift[j] + 97)
    # decode text with new key
    decoded_text = decode_vigenere(key, text=text)
    # with hack_caesar find first letter for correct key
    step = hack_caesar(model, text=decoded_text)
    # calculate correct key
    key = str(chr(step + 97))
    for i in range(len(shift)):
        key += chr((step + shift[i]) % 26 + 97)
    # decode text with new key
    decode_vigenere(key, input_file, output_file)
    return key


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='List of commands')

    # encode command parser
    encode_parser = subparsers.add_parser('encode', help='Encode input message')
    encode_parser.add_argument('-c', '--cipher', required=True, type=str,
                               help='Type of cipher: Caesar or Vigenere or Vernam')
    encode_parser.add_argument('-k', '--key', required=True, type=str, help='Cipher key')
    encode_parser.add_argument('--input', type=str, dest='input_file', help='Path to the input file')
    encode_parser.add_argument('--output', type=str, dest='output_file', help='Path to the output file')
    encode_parser.set_defaults(func=encode)

    # decode command parser
    decode_parser = subparsers.add_parser('decode', help='Decode input message')
    decode_parser.add_argument('-c', '--cipher', required=True, type=str,
                               help='Type of cipher: Caesar or Vigenere or Vernam')
    decode_parser.add_argument('-k', '--key', required=True, type=str, help='Cipher key')
    decode_parser.add_argument('--input', type=str, dest='input_file', help='Path to the input file')
    decode_parser.add_argument('--output', type=str, dest='output_file', help='Path to the output file')
    decode_parser.set_defaults(func=decode)

    # train command parser
    train_parser = subparsers.add_parser('train', help='Train language model on given text(Only for Caesar cipher)')
    train_parser.add_argument('-m', '--model', required=True, type=str,
                              dest='model_file', help='File where language model will be written')
    train_parser.add_argument('--input', type=str, dest='input_file', help='Path to the input file')
    train_parser.set_defaults(func=train)

    # hack command parser
    hack_parser = subparsers.add_parser('hack', help='Try to hack Caesar or Vinegere code')
    hack_parser.add_argument('-c', '--cipher', required=True, type=str,
                             help='Type of cipher: Caesar or Vigenere or Vernam')
    hack_parser.add_argument('-m', '--model', required=True, type=str,
                             dest='model_file', help='File where language model will be taken from')
    hack_parser.add_argument('--input', type=str, dest='input_file', help='Path to the input file')
    hack_parser.add_argument('--output', type=str, dest='output_file', help='Path to the output file')
    hack_parser.set_defaults(func=hack)

    # accuracy test for hack caesar
    def test_accuracy():
        args = parser.parse_args('train --input shakespeare.txt --model model.json'.split())
        args.func(args)

        right = ''
        for i in range(26):
            args = parser.parse_args(
                'encode --cipher caesar --key {} --input big.txt --output output.txt'.format(i).split())
            args.func(args)

            args = parser.parse_args('hack --cipher caesar --input output.txt --model model.json --output trash.txt'.split())
            ans = args.func(args)

            if ans == i:
                right += ' ' + str(i)
        print(len(right.split()) / 26, right)
        return

    # test for vigenere hack
    def test_vigenere_hack():
        args = parser.parse_args(
            'encode --cipher vigenere --key kekolusik --input input.txt --output output.txt'.split())
        args.func(args)
        args = parser.parse_args('hack --cipher vigenere --model model.json'
                                 ' --input output.txt --output trash.txt'.split())
        key = args.func(args)
        print(key)

    # test vernam encode and decode
    def test_encode_and_decode():
        args = parser.parse_args('encode --cipher vernam --key ABADBDFNSNERNESNDSFB '
                                 '--input short_inp.txt --output output.txt'.split())
        args.func(args)
        args = parser.parse_args('decode --cipher vernam --key ABADBDFNSNERNESNDSFB '
                                 '--input output.txt'.split())
        args.func(args)
        return

    # testing block
    # test_vigenere_hack()
    # test_encode_and_decode()
    # test_accuracy()

    args = parser.parse_args()
    if 'func' not in vars(args):
        raise SyntaxError('No functions')
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
