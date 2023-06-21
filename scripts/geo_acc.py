import pandas as pd
from sklearn.metrics import accuracy_score

CJK = ['yue', 'ja', 'ko', 'zh-CN']
CMN = ['ar', 'az', 'he', 'kk', 'ky', 'mn', 'ps', 'fa', 'ckb', 'tg', 'tr', 'uz']
EE = ['hy', 'be', 'bg', 'cs', 'et', 'ka', 'lv', 'lt', 'mk', 'pl', 'ro', 'ru', 'sr', 'sk', 'sl', 'uk']
SA = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'sd', 'ta', 'te', 'ur']
SEA = ['my', 'ceb', 'tl', 'id', 'jv', 'km', 'lo', 'ms', 'mi', 'th', 'vi']
SSA = ['af', 'am', 'ff', 'lg', 'ha', 'ig', 'kam', 'ln', 'luo', 'nso', 'ny', 'om', 'sn', 'so', 'sw', 'umb', 'wo', 'xh', 'yo', 'zu']
WE = ['ast', 'bs', 'ca', 'hr', 'da', 'nl', 'en', 'fi', 'fr', 'gl', 'de', 'el', 'hu', 'is', 'ga', 'it', 'kea', 'lb', 'mt', 'no', 'oc', 'pt', 'es', 'sv', 'cy']

preds = pd.read_csv('/home/itk0123/x-vector-pytorch/outputs/0613_1720_output_oldfilt_16_fleurs/preds.csv', index_col=0)
# print(preds)

# for A in [CJK, CMN, EE, SA, SEA, SSA, WE]:
# with CJK as A:

def get_ids(labels):
    ids = []
    for r in preds.loc[:,'Ground Truth'].items():
        if r[1] in labels:
            ids.append(r[0])
    return ids

for A in [WE, EE, CMN, SSA, SA, SEA, CJK]:
    print(accuracy_score(preds.iloc[get_ids(A),1],preds.iloc[get_ids(A),0]), end=',')