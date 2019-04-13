import encryptor


# accuracy test for hack caesar
def test_accuracy():
    args = encryptor.parse_args('train --text-file shakespeare.txt --model-file model.json'.split())
    args.func(args)
    right = []
    for i in range(5):
        args = encryptor.parse_args(
            'encode --cipher caesar --key {} --input-file big.txt --output-file output.txt'.format(i).split())
        args.func(args)
        args = encryptor.parse_args('hack --input-file output.txt --model-file model.json '
                                    '--output-file trash.txt'.split())
        ans = args.func(args)

        if ans == i:
            right += str(i)
    print(len(right) / 5, ' '.join(right))


# test for vigenere hack
def test_vigenere_hack():
    args = encryptor.parse_args(
        'encode --cipher vigenere --key kekolusik --input-file input.txt --output-file output.txt'.split())
    args.func(args)
    args = encryptor.parse_args('hack --cipher vigenere --model-file model.json'
                                ' --input-file output.txt --output-file trash.txt'.split())
    key = args.func(args)
    print(key)


# test vernam encode and decode
def test_encode_and_decode():
    args = encryptor.parse_args('encode --cipher vernam --key ABADBDFNSNERNESNDSFB '
                                '--input-file short_inp.txt --output-file output.txt'.split())
    args.func(args)
    args = encryptor.parse_args('decode --cipher vernam --key ABADBDFNSNERNESNDSFB '
                                '--input-file output.txt'.split())
    args.func(args)


def main():
    test_vigenere_hack()
    test_encode_and_decode()
    test_accuracy()


if __name__ == '__main__':
    main()
