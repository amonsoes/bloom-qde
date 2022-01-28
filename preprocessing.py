from claque import QuestionMasker
import argparse


if __name__=='__main__':
    '''
        Masks a file with POS tags
        Input: csv file that is separated by semicolon
        Output: csv file with POS tags and classes (EASY, DIFFICULT)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='./data/unmasked_test_data_binary.csv',
                        help='input unmasked data file')
    parser.add_argument('--outfile', type=str, default='./data/test_data_binary.csv',
                        help='output test data file')
    FLAGS, unparsed = parser.parse_known_args()
    
    qm = QuestionMasker()
    qm.mask_file(FLAGS.infile, FLAGS.outfile)
