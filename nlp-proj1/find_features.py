from features import \
    load_df, load_lex_conno, load_lex_vad, load_user_data, eval_big_issues_dict

from better_profanity import profanity
from nltk import word_tokenize
import numpy as np

profanity.load_censor_words()
SWEAR_WORDS = set(str(s) for s in profanity.CENSOR_WORDSET)

MODAL_VERBS = {
    'can', 'could', 'may', 'might', 'will', 'would', 'shall', 'should', 'must', 'ought', 'dare'
}

PRONOUNS_FIRST = {
    'I', 'we', 'me', 'us', 'my', 'mine', 'our', 'ours', 'myself', 'ourselves'
}

PRONOUNS_SECOND = {
    'you', 'your', 'yours', 'yourself', 'yourselves'
}

PRONOUNS_THIRD = {
    'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'hers', 'its', 'their', 'theirs',
    'himself', 'herself', 'themselves', 'itself'
}

GENDERS = {
    'Agender': 0,
    'Androgyne': 1,
    'Bigender': 2,
    'Female': 3,
    'Genderqueer': 4,
    'Male': 5,
    'Prefer not to say': 6,
    'Transgender Female': 7,
    'Transgender Male': 8
}


def cossim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


BM1 = np.array((0.1809, 0.4309, 0.5426, 0.5053, 0.3936, 0.5319, 0.4096, 0.3883, 0.3138, 0.516, 0.4096, 0.383, 0.5213, 0.3191, 0.617, 0.4149, 0.5266, 0.2606, 0.3723, 0.5426, 0.5904, 0.3883, 0.4894, 0.5319, 0.4362, 0.4255, 0.4521, 0.3404, 0.4309, 0.4947, 0.3723, 0.3138, 0.4681, 0.5426, 0.3351, 0.3989, 0.5745, 0.3191, 0.3404, 0.3404, 0.5213, 0.5745, 0.3883, 0.4894, 0.3138, 0.4202, 0.3617, 0.3404), dtype=np.float32)
BM2 = np.array((0.2085, 0.3697, 0.4976, 0.5166, 0.3697, 0.5024, 0.4076, 0.3412, 0.2749, 0.5498, 0.4028, 0.3981, 0.5687, 0.3602, 0.6161, 0.3602, 0.4976, 0.2559, 0.3839, 0.5782, 0.5687, 0.3744, 0.4929, 0.5213, 0.3934, 0.3223, 0.4123, 0.3412, 0.3981, 0.4834, 0.3555, 0.3318, 0.3839, 0.5545, 0.3175, 0.4502, 0.4976, 0.2938, 0.3128, 0.3365, 0.4787, 0.5829, 0.4123, 0.4787, 0.4028, 0.3507, 0.3697, 0.3033), dtype=np.float32)


def gen_feats(df):
    lex_conno = load_lex_conno('lexica/connotation_lexicon_a.0.1.csv')
    lex_vad = load_lex_vad('lexica/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt')
    user_data = load_user_data('data/users.json')
    n = len(df)

    # Lex features
    conno = np.zeros((n, 3), dtype=np.float32)
    vad_pro = np.zeros((n, 3), dtype=np.float32)
    vad_con = np.zeros((n, 3), dtype=np.float32)
    vad = np.zeros((n, 3), dtype=np.float32)

    # Ling features
    length = np.zeros((n, 1), dtype=np.float32)
    modal_verb = np.zeros((n, 1), dtype=np.float32)
    exclaim = np.zeros((n, 1), dtype=np.float32)
    swear = np.zeros((n, 1), dtype=np.float32)
    pronoun = np.zeros((n, 3), dtype=np.float32)

    # User features
    big_issues = np.zeros((n, 48), dtype=np.float32)
    politic = np.zeros((n, 1), dtype=np.float32)
    friend = np.zeros((n, 1), dtype=np.float32)

    # Label
    label = np.zeros(n, dtype=np.int)

    for i, (_, row) in enumerate(df.iterrows()):
        doc_vad_pro, doc_vad_con = [], []
        n_pro = n_con = 0
        pro, con = row['pro_debater'], row['con_debater']
        pro_info, con_info = user_data[pro], user_data[con]
        doc_conno_pro, doc_conno_con = np.zeros(3), np.zeros(3)
        len_pro = len_con = 0
        modal_verb_pro = modal_verb_con = 0
        exclaim_pro = exclaim_con = 0
        swear_pro = swear_con = 0
        pronoun_pro, pronoun_con = np.zeros(3), np.zeros(3)

        # User features
        big_issues[i] = \
            eval_big_issues_dict(pro_info['big_issues_dict']) == eval_big_issues_dict(con_info['big_issues_dict'])
        politic[i] = pro_info['political_ideology'] == con_info['political_ideology']
        friend[i] = pro in con_info['friends'] and con in pro_info['friends']

        for debate_round in row['rounds']:
            for side in debate_round:
                text = side['text']
                tokens = word_tokenize(text.lower())
                if side['side'] == 'Pro':
                    n_pro += 1
                    len_pro += len(tokens)
                    for token in tokens:
                        if token in lex_conno:
                            doc_conno_pro[lex_conno[token]] += 1
                        if token in lex_vad:
                            doc_vad_pro.append(lex_vad[token])
                        if token in MODAL_VERBS:
                            modal_verb_pro += 1
                        if token == '!':
                            exclaim_pro += 1
                        if token in SWEAR_WORDS:
                            swear_pro += 1
                        if token in PRONOUNS_FIRST:
                            pronoun_pro[0] += 1
                        if token in PRONOUNS_SECOND:
                            pronoun_pro[1] += 1
                        if token in PRONOUNS_THIRD:
                            pronoun_pro[2] += 1
                else:
                    n_con += 1
                    len_con += len(tokens)
                    for token in tokens:
                        if token in lex_conno:
                            doc_conno_con[lex_conno[token]] += 1
                        if token in lex_vad:
                            doc_vad_con.append(lex_vad[token])
                        if token in MODAL_VERBS:
                            modal_verb_con += 1
                        if token == '!':
                            exclaim_con += 1
                        if token in SWEAR_WORDS:
                            swear_con += 1
                        if token in PRONOUNS_FIRST:
                            pronoun_con[0] += 1
                        if token in PRONOUNS_SECOND:
                            pronoun_con[1] += 1
                        if token in PRONOUNS_THIRD:
                            pronoun_con[2] += 1

        # Lex features
        conno[i] = doc_conno_pro > doc_conno_con
        if len(doc_vad_pro):
            vad_pro[i] = np.asarray(doc_vad_pro, dtype=np.float32).mean(axis=0)
        if len(doc_vad_con):
            vad_con[i] = np.asarray(doc_vad_con, dtype=np.float32).mean(axis=0)
        vad[i] = vad_pro[i] > vad_con[i]

        # Ling features
        length[i] = len_pro > len_con
        modal_verb[i] = modal_verb_pro > modal_verb_con
        exclaim[i] = exclaim_pro > exclaim_con
        swear[i] = swear_pro > swear_con
        pronoun[i] = pronoun_pro > pronoun_con
        # print(row['id'], row['winner'], np.linalg.norm(big_issues[i] - bm0), np.linalg.norm(big_issues[i] - bm1), cossim(big_issues[i], bm0), cossim(big_issues[i], bm1), politic[i])

        label[i] = (row['winner'] == 'Con')  # Pro 0, Con 1

    return {
               'conno': conno,
               'vad': np.concatenate((vad_pro, vad_con), axis=1),
               'length': length,
               'modal_verb': modal_verb,
               'exclaim': exclaim,
               'swear': swear,
               'pronoun': pronoun,
               'big_issues': big_issues,
               'politic': politic,
               'friend': friend,
           }, label


def main():
    df, df_r, df_nr = load_df('data/val.jsonl')
    print(sum(df_r['winner'] == 'Con') / len(df_r))
    print(sum(df_nr['winner'] == 'Con') / len(df_nr))

    feats, y = gen_feats(df_nr)

    feat: np.ndarray = feats['friend']
    feat0, feat1 = feat[y == 0], feat[y == 1]
    mean0, std0 = feat0.mean(axis=0), feat0.std(axis=0)
    mean1, std1 = feat1.mean(axis=0), feat1.std(axis=0)
    print('pro:', tuple(np.round(mean0, 4)), tuple(np.round(std0, 4)))
    print('con:', tuple(np.round(mean1, 4)), tuple(np.round(std1, 4)))
    print(cossim(mean0, mean1))


if __name__ == '__main__':
    main()
